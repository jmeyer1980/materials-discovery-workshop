"""
Materials Discovery ML Model

This module implements machine learning models for materials discovery,
including variational autoencoders for generating new alloy compositions.

CORE METHODOLOGY ATTRIBUTION:
- Variational Autoencoder (VAE) architecture based on:
  Kingma, D. P., & Welling, M. (2013).
  "Auto-Encoding Variational Bayes."
  arXiv preprint arXiv:1312.6114
  https://arxiv.org/abs/1312.6114

DEPENDENCIES:
- PyTorch (Paszke et al., 2019): Deep learning framework
- Scikit-learn (Pedregosa et al., 2011): ML algorithms and preprocessing
- Pymatgen (Ong et al., 2013): Materials property data and structures
- Pandas (McKinney, 2010): Data manipulation
- NumPy (Harris et al., 2020): Numerical computing

LICENSE: MIT (see LICENSE file in repository)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random


class MaterialsDataset(Dataset):
    """PyTorch Dataset for materials data."""

    def __init__(self, data: pd.DataFrame, feature_columns: List[str], target_columns: Optional[List[str]] = None):
        """Initialize the dataset.

        Args:
            data: DataFrame containing materials data
            feature_columns: List of column names to use as features
            target_columns: List of column names to use as targets (optional)
        """
        self.data = data.copy()
        self.feature_columns = feature_columns
        self.target_columns = target_columns

        # Extract features
        self.features = self.data[feature_columns].values.astype(np.float32)

        # Extract targets if provided
        if target_columns:
            self.targets = self.data[target_columns].values.astype(np.float32)
        else:
            self.targets = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.features[idx]

        if self.targets is not None:
            targets = self.targets[idx]
            return features, targets
        else:
            return features,  # Return tuple even for single item to maintain consistency


class FeatureEngineer:
    """Feature engineering for materials data."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.scalers = {}

    def create_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create feature vectors from materials data.

        Args:
            data: Raw materials DataFrame

        Returns:
            Tuple of (processed_data, feature_columns)
        """
        processed_data = data.copy()

        # Fill NaN values for ternary alloys (element_3 and composition_3 are NaN for binary)
        processed_data['element_3'] = processed_data['element_3'].fillna('None')
        processed_data['composition_3'] = processed_data['composition_3'].fillna(0.0)

        # Create one-hot encoded features for elements
        element_cols = ['element_1', 'element_2', 'element_3']

        # Get unique elements (including 'None' for binary alloys)
        all_elements = set()
        for col in element_cols:
            all_elements.update(processed_data[col].unique())
        all_elements = sorted(list(all_elements))

        # One-hot encode elements
        for i, col in enumerate(element_cols):
            for elem in all_elements:
                processed_data[f'{col}_{elem}'] = (processed_data[col] == elem).astype(int)

        # Composition features
        composition_features = ['composition_1', 'composition_2', 'composition_3']

        # Interaction features (products of compositions)
        processed_data['comp_1_2'] = processed_data['composition_1'] * processed_data['composition_2']
        processed_data['comp_1_3'] = processed_data['composition_1'] * processed_data['composition_3']
        processed_data['comp_2_3'] = processed_data['composition_2'] * processed_data['composition_3']

        # Property-based features
        processed_data['mp_bp_ratio'] = processed_data['melting_point'] / (processed_data['boiling_point'] + 1e-6)
        processed_data['electroneg_sq'] = processed_data['electronegativity'] ** 2
        processed_data['radius_sq'] = processed_data['atomic_radius'] ** 2

        # Define feature columns
        element_features = [f'{col}_{elem}' for col in element_cols for elem in all_elements]
        composition_features = ['composition_1', 'composition_2', 'composition_3',
                               'comp_1_2', 'comp_1_3', 'comp_2_3']
        property_features = ['melting_point', 'boiling_point', 'density',
                           'electronegativity', 'atomic_radius', 'electronegativity_difference',
                           'mp_bp_ratio', 'electroneg_sq', 'radius_sq']

        feature_columns = element_features + composition_features + property_features

        return processed_data, feature_columns

    def scale_features(self, data: pd.DataFrame, feature_columns: List[str],
                      fit: bool = True) -> pd.DataFrame:
        """Scale features using StandardScaler.

        Args:
            data: DataFrame with features
            feature_columns: List of feature column names
            fit: Whether to fit the scaler (True for training, False for inference)

        Returns:
            DataFrame with scaled features
        """
        scaled_data = data.copy()

        if fit:
            self.scalers['features'] = StandardScaler()
            scaled_features = self.scalers['features'].fit_transform(data[feature_columns])
        else:
            if 'features' not in self.scalers:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            scaled_features = self.scalers['features'].transform(data[feature_columns])

        scaled_data[feature_columns] = scaled_features
        return scaled_data


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder for materials generation."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: List[int] = [128, 64]):
        """Initialize the VAE.

        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space
            hidden_dims: Dimensions of hidden layers in encoder/decoder
        """
        super(VariationalAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim

        # Latent space
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        current_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            current_dim = hidden_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())  # Assuming features are normalized to [0,1]

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """Encode input to latent space."""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode from latent space."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var

    def sample(self, num_samples: int, device: str = 'cpu'):
        """Sample from the latent space."""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decode(z)
            return samples


class MaterialsDiscoveryModel:
    """Main model for materials discovery."""

    def __init__(self, latent_dim: int = 32, learning_rate: float = 1e-3,
                 batch_size: int = 64, epochs: int = 100):
        """Initialize the materials discovery model.

        Args:
            latent_dim: Dimension of latent space
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.model = None
        self.feature_engineer = FeatureEngineer()
        self.optimizer = None
        self.feature_columns = None

    def _vae_loss(self, reconstructed, original, mu, log_var):
        """Compute VAE loss (reconstruction + KL divergence)."""
        # Reconstruction loss
        reconstruction_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum')

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return reconstruction_loss + kl_loss

    def train(self, data_path: str):
        """Train the VAE model on materials data.

        Args:
            data_path: Path to CSV file containing materials data
        """
        # Load and preprocess data
        print("Loading data...")
        data = pd.read_csv(data_path)

        print("Creating features...")
        data_processed, self.feature_columns = self.feature_engineer.create_features(data)

        print("Scaling features...")
        data_scaled = self.feature_engineer.scale_features(data_processed, self.feature_columns, fit=True)

        # Scale to [0,1] for sigmoid activation
        minmax_scaler = MinMaxScaler()
        data_scaled[self.feature_columns] = minmax_scaler.fit_transform(data_scaled[self.feature_columns])
        self.minmax_scaler = minmax_scaler

        # Create dataset
        dataset = MaterialsDataset(data_scaled, self.feature_columns)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        input_dim = len(self.feature_columns)
        self.model = VariationalAutoencoder(input_dim, self.latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f"Training VAE with {input_dim} input dimensions, {self.latent_dim} latent dimensions...")

        # Training loop
        self.model.train()
        losses = []

        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_features, in dataloader:
                batch_features = batch_features.to(self.device)

                # Forward pass
                reconstructed, mu, log_var = self.model(batch_features)

                # Compute loss
                loss = self._vae_loss(reconstructed, batch_features, mu, log_var)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

        print("Training completed!")

        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('VAE Training Loss')
        plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()

        return losses

    def generate_new_materials(self, num_samples: int = 100,
                              target_properties: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
        """Generate new materials using the trained VAE.

        Args:
            num_samples: Number of new materials to generate
            target_properties: Optional constraints on properties (property_name -> (min, max))

        Returns:
            DataFrame with generated materials
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        self.model.eval()

        print(f"Generating {num_samples} new materials...")

        # Sample from latent space
        generated_features = self.model.sample(num_samples, self.device).cpu().numpy()

        # Inverse transform to original scale
        generated_features = self.minmax_scaler.inverse_transform(generated_features)
        generated_features = self.feature_engineer.scalers['features'].inverse_transform(generated_features)

        # Create DataFrame with generated features
        generated_data = pd.DataFrame(generated_features, columns=self.feature_columns)

        # Post-process generated materials
        processed_materials = self._post_process_generated_materials(generated_data, target_properties)

        return processed_materials

    def _post_process_generated_materials(self, generated_data: pd.DataFrame,
                                        target_properties: Optional[Dict[str, Tuple[float, float]]] = None) -> pd.DataFrame:
        """Post-process generated materials to create interpretable alloy formulas.

        Args:
            generated_data: Raw generated feature data
            target_properties: Optional property constraints

        Returns:
            Processed DataFrame with alloy formulas and properties
        """
        processed_data = []

        for idx, row in generated_data.iterrows():
            # Extract element probabilities from one-hot encoded features
            element_probs = self._extract_element_probabilities(row)

            # Select elements based on probabilities
            selected_elements = self._select_elements_from_probabilities(element_probs)

            if not selected_elements:
                continue  # Skip if no valid elements selected

            # Generate compositions
            compositions = self._generate_compositions(selected_elements, row)

            # Calculate properties
            properties = self._calculate_generated_properties(selected_elements, compositions, row)

            # Apply property constraints if specified
            if target_properties:
                if not self._satisfies_constraints(properties, target_properties):
                    continue

            # Create alloy formula
            formula = self._create_formula_string(selected_elements, compositions)

            material_data = {
                'formula': formula,
                'elements': selected_elements,
                'compositions': compositions,
                **properties,
                'is_generated': True
            }

            processed_data.append(material_data)

        return pd.DataFrame(processed_data)

    def _extract_element_probabilities(self, row) -> Dict[str, float]:
        """Extract element probabilities from generated features."""
        element_probs = {}

        # Get all element columns (element_1_*, element_2_*, element_3_*)
        element_columns = [col for col in self.feature_columns if col.startswith('element_')]

        # Group by element position
        for pos in ['1', '2', '3']:
            pos_cols = [col for col in element_columns if f'element_{pos}_' in col]
            pos_probs = {}

            for col in pos_cols:
                element = col.replace(f'element_{pos}_', '')
                prob = max(0, min(1, row[col]))  # Clamp to [0,1]
                pos_probs[element] = prob

            element_probs[pos] = pos_probs

        return element_probs

    def _select_elements_from_probabilities(self, element_probs: Dict[str, Dict[str, float]]) -> List[str]:
        """Select elements based on generated probabilities."""
        selected_elements = []

        for pos in ['1', '2', '3']:
            probs = element_probs[pos]

            # Remove 'None' and normalize probabilities
            valid_probs = {k: v for k, v in probs.items() if k != 'None' and v > 0.1}
            if not valid_probs:
                continue

            # Normalize probabilities
            total_prob = sum(valid_probs.values())
            if total_prob > 0:
                valid_probs = {k: v/total_prob for k, v in valid_probs.items()}

                # Sample element based on probabilities
                elements = list(valid_probs.keys())
                probabilities = list(valid_probs.values())

                selected_element = np.random.choice(elements, p=probabilities)
                if selected_element not in selected_elements:
                    selected_elements.append(selected_element)

        return selected_elements[:3]  # Limit to ternary at most

    def _generate_compositions(self, elements: List[str], row) -> List[float]:
        """Generate compositions for selected elements."""
        n_elements = len(elements)

        if n_elements == 0:
            return []

        # Use composition features from generated data
        comp_features = ['composition_1', 'composition_2', 'composition_3']

        compositions = []
        for i in range(n_elements):
            if i < len(comp_features):
                comp = max(0.05, min(0.95, row[comp_features[i]]))  # Clamp to reasonable range
                compositions.append(comp)
            else:
                compositions.append(0.33)  # Default for ternary

        # Normalize to sum to 1
        total = sum(compositions)
        if total > 0:
            compositions = [c/total for c in compositions]

        return compositions

    def _calculate_generated_properties(self, elements: List[str], compositions: List[float],
                                     row) -> Dict[str, float]:
        """Calculate properties for generated material."""
        properties = {}

        # Use generated property values directly (they're already in the features)
        property_mapping = {
            'melting_point': 'melting_point',
            'boiling_point': 'boiling_point',
            'density': 'density',
            'electronegativity': 'electronegativity',
            'atomic_radius': 'atomic_radius'
        }

        for prop_name, feature_name in property_mapping.items():
            if feature_name in row:
                properties[prop_name] = float(row[feature_name])

        # Calculate electronegativity difference for binary alloys
        if len(elements) == 2:
            # This would need element property lookup - for now use generated value
            properties['electronegativity_difference'] = float(row.get('electronegativity_difference', 0))

        return properties

    def _satisfies_constraints(self, properties: Dict[str, float],
                             constraints: Dict[str, Tuple[float, float]]) -> bool:
        """Check if generated material satisfies property constraints."""
        for prop_name, (min_val, max_val) in constraints.items():
            if prop_name in properties:
                prop_val = properties[prop_name]
                if prop_val < min_val or prop_val > max_val:
                    return False
        return True

    def _create_formula_string(self, elements: List[str], compositions: List[float]) -> str:
        """Create chemical formula string from elements and compositions."""
        if len(elements) != len(compositions):
            return "Invalid"

        formula_parts = []
        for elem, comp in zip(elements, compositions):
            formula_parts.append(f"{elem}{comp:.3f}")

        return "".join(formula_parts)


def main():
    """Train the model and generate new materials."""
    # Initialize model
    model = MaterialsDiscoveryModel(latent_dim=32, epochs=50, batch_size=64)

    # Train on dataset
    losses = model.train('materials_dataset.csv')

    # Generate new materials
    print("\nGenerating new materials...")
    new_materials = model.generate_new_materials(num_samples=20)

    print(f"\nGenerated {len(new_materials)} new materials:")
    if len(new_materials) > 0:
        print(new_materials.head(10)[['formula', 'melting_point', 'density']].to_string())

        # Save generated materials
        new_materials.to_csv('generated_materials.csv', index=False)
        print("\nSaved generated materials to 'generated_materials.csv'")
    else:
        print("No valid materials generated.")


if __name__ == "__main__":
    main()
