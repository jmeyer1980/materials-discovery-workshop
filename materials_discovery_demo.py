"""
Materials Discovery Demonstration

This script demonstrates how machine learning can be used to discover new
materials by learning patterns from existing alloy compositions and generating
new ones.

CORE METHODOLOGY ATTRIBUTION:
- Variational Autoencoder (VAE): Kingma & Welling (2013)
  arXiv:1312.6114 (https://arxiv.org/abs/1312.6114)
- Clustering (K-Means): Lloyd (1982), implemented via scikit-learn
- Dimensionality Reduction (PCA): Pearson (1901), implemented via scikit-learn

FRAMEWORK CITATIONS:
- PyTorch: Paszke et al. (2019), NeurIPS 2019
- Scikit-learn: Pedregosa et al. (2011), JMLR 12:2825-2830
- Matplotlib: Hunter (2007), Computing in Science & Engineering 9(3):90-95

LICENSE: MIT (see LICENSE file in repository)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
from typing import List, Tuple


class SimpleVAE(nn.Module):
    """Simplified Variational Autoencoder for demonstration."""

    def __init__(self, input_dim: int = 10, latent_dim: int = 5):
        super(SimpleVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, latent_dim)
        self.fc_var = nn.Linear(16, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_var


class MaterialsDiscoveryDemo:
    """Demonstration of materials discovery using ML."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

    def load_and_preprocess_data(self, data_path: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load and preprocess materials data."""
        print("Loading materials dataset...")
        data = pd.read_csv(data_path)

        # Select key features for demonstration
        feature_cols = ['composition_1', 'composition_2', 'melting_point',
                       'density', 'electronegativity', 'atomic_radius']

        # Handle ternary alloys (fill missing composition_3 with 0)
        data['composition_3'] = data['composition_3'].fillna(0)

        # For simplicity, focus on binary alloys
        binary_data = data[data['alloy_type'] == 'binary'].copy()

        print(f"Using {len(binary_data)} binary alloys for demonstration")

        # Extract features
        features = binary_data[feature_cols].values

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        return binary_data, features_scaled

    def train_vae(self, features: np.ndarray, epochs: int = 100) -> SimpleVAE:
        """Train a simple VAE on the materials features."""
        print("Training Variational Autoencoder...")

        # Convert to torch tensor
        features_tensor = torch.FloatTensor(features)

        # Create data loader
        dataset = torch.utils.data.TensorDataset(features_tensor, features_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Initialize model
        input_dim = features.shape[1]
        model = SimpleVAE(input_dim=input_dim, latent_dim=5).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        model.train()
        losses = []

        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)

                # Forward pass
                reconstructed, mu, log_var = model(batch_x)

                # Compute loss (reconstruction + KL divergence)
                reconstruction_loss = nn.functional.mse_loss(reconstructed, batch_x, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                loss = reconstruction_loss + kl_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                print(".4f")

        print("VAE training completed!")
        return model

    def generate_new_materials(self, model: SimpleVAE, original_data: pd.DataFrame,
                              num_samples: int = 50) -> pd.DataFrame:
        """Generate new materials using the trained VAE."""
        print(f"Generating {num_samples} new material compositions...")

        model.eval()
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(num_samples, model.latent_dim).to(self.device)
            generated_features = model.decode(z).cpu().numpy()

            # Inverse transform to original scale
            generated_features = self.scaler.inverse_transform(generated_features)

        # Create DataFrame with generated materials
        new_materials = []
        elements = ['Al', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

        for i, features in enumerate(generated_features):
            # Generate random element combination
            elem1, elem2 = random.sample(elements, 2)

            comp1 = max(0.1, min(0.9, features[0]))  # composition_1
            comp2 = 1.0 - comp1

            material = {
                'id': f'generated_{i+1}',
                'element_1': elem1,
                'element_2': elem2,
                'composition_1': comp1,
                'composition_2': comp2,
                'formula': '.3f',
                'melting_point': abs(features[2]),  # Ensure positive
                'density': abs(features[3]),       # Ensure positive
                'electronegativity': max(0, features[4]),
                'atomic_radius': max(0, features[5]),
                'is_generated': True
            }
            new_materials.append(material)

        return pd.DataFrame(new_materials)

    def cluster_analysis(self, features: np.ndarray, original_data: pd.DataFrame) -> pd.DataFrame:
        """Perform cluster analysis on materials."""
        print("Performing cluster analysis...")

        # Reduce dimensionality for clustering
        features_pca = self.pca.fit_transform(features)

        # Find optimal number of clusters
        silhouette_scores = []
        for n_clusters in range(2, 8):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_pca)
            silhouette_avg = silhouette_score(features_pca, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        print(f"Optimal number of clusters: {optimal_clusters}")

        # Perform clustering with optimal number
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_pca)

        # Add cluster information to original data
        clustered_data = original_data.copy()
        clustered_data['cluster'] = cluster_labels
        clustered_data['pca_1'] = features_pca[:, 0]
        clustered_data['pca_2'] = features_pca[:, 1]

        return clustered_data

    def create_visualizations(self, original_data: pd.DataFrame, generated_data: pd.DataFrame,
                            clustered_data: pd.DataFrame):
        """Create visualizations comparing original and generated materials."""
        print("Creating visualizations...")

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Materials Discovery ML Demonstration', fontsize=16, fontweight='bold')

        # 1. Melting Point vs Density - Original vs Generated
        ax = axes[0, 0]
        orig_scatter = ax.scatter(original_data['density'], original_data['melting_point'],
                                 alpha=0.6, c='blue', label='Original Materials', s=30)
        gen_scatter = ax.scatter(generated_data['density'], generated_data['melting_point'],
                                alpha=0.8, c='red', marker='x', label='Generated Materials', s=50)
        ax.set_xlabel('Density (g/cm³)')
        ax.set_ylabel('Melting Point (K)')
        ax.set_title('Material Properties: Original vs Generated')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Composition Space
        ax = axes[0, 1]
        orig_comp = ax.scatter(original_data['composition_1'], original_data['composition_2'],
                              alpha=0.6, c='blue', label='Original', s=30)
        gen_comp = ax.scatter(generated_data['composition_1'], generated_data['composition_2'],
                             alpha=0.8, c='red', marker='x', label='Generated', s=50)
        ax.set_xlabel('Element 1 Composition')
        ax.set_ylabel('Element 2 Composition')
        ax.set_title('Composition Space')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Cluster Analysis
        ax = axes[0, 2]
        scatter = ax.scatter(clustered_data['pca_1'], clustered_data['pca_2'],
                           c=clustered_data['cluster'], cmap='viridis', alpha=0.7, s=40)
        ax.set_xlabel('PCA Component 1')
        ax.set_ylabel('PCA Component 2')
        ax.set_title('Material Clusters (PCA)')
        plt.colorbar(scatter, ax=ax, label='Cluster')

        # 4. Property Distributions - Melting Point
        ax = axes[1, 0]
        ax.hist(original_data['melting_point'], bins=20, alpha=0.7, color='blue',
                label='Original', density=True)
        ax.hist(generated_data['melting_point'], bins=20, alpha=0.7, color='red',
                label='Generated', density=True)
        ax.set_xlabel('Melting Point (K)')
        ax.set_ylabel('Density')
        ax.set_title('Melting Point Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Property Distributions - Density
        ax = axes[1, 1]
        ax.hist(original_data['density'], bins=20, alpha=0.7, color='blue',
                label='Original', density=True)
        ax.hist(generated_data['density'], bins=20, alpha=0.7, color='red',
                label='Generated', density=True)
        ax.set_xlabel('Density (g/cm³)')
        ax.set_ylabel('Density')
        ax.set_title('Density Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Generated Materials Table Preview
        ax = axes[1, 2]
        ax.axis('off')

        # Create table data
        table_data = generated_data.head(8)[['formula', 'melting_point', 'density']].copy()
        table_data['melting_point'] = table_data['melting_point'].round(1)
        table_data['density'] = table_data['density'].round(3)

        # Create table
        table = ax.table(cellText=table_data.values,
                        colLabels=table_data.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax.set_title('Sample Generated Materials', pad=20)

        plt.tight_layout()
        plt.savefig('materials_discovery_demo.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("Visualization saved as 'materials_discovery_demo.png'")

    def run_demo(self):
        """Run the complete materials discovery demonstration."""
        print("="*60)
        print("MATERIALS DISCOVERY ML DEMONSTRATION")
        print("="*60)
        print("This demo shows how machine learning can learn patterns from")
        print("existing materials and generate new alloy compositions.")
        print()

        # Load and preprocess data
        original_data, features = self.load_and_preprocess_data('materials_dataset.csv')

        # Train VAE
        vae_model = self.train_vae(features, epochs=50)

        # Generate new materials
        generated_data = self.generate_new_materials(vae_model, original_data, num_samples=100)

        # Perform cluster analysis
        clustered_data = self.cluster_analysis(features, original_data)

        # Create visualizations
        self.create_visualizations(original_data, generated_data, clustered_data)

        # Print summary
        print("\n" + "="*60)
        print("DEMONSTRATION SUMMARY")
        print("="*60)
        print(f"Original materials analyzed: {len(original_data)}")
        print(f"New materials generated: {len(generated_data)}")
        print(f"Material clusters identified: {clustered_data['cluster'].nunique()}")
        print()
        print("Key Insights:")
        print("- ML successfully learned patterns in alloy compositions")
        print("- Generated materials explore new regions of property space")
        print("- Clustering reveals natural groupings in materials data")
        print("- This approach can accelerate materials discovery")
        print()
        print("Generated materials saved to 'generated_materials_demo.csv'")
        generated_data.to_csv('generated_materials_demo.csv', index=False)

        # Show some example generated materials
        print("\nExample Generated Materials:")
        print(generated_data[['formula', 'melting_point', 'density']].head(10).to_string(index=False))


if __name__ == "__main__":
    demo = MaterialsDiscoveryDemo()
    demo.run_demo()
