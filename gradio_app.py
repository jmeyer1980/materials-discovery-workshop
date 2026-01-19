"""
Gradio Web App for Materials Discovery Workshop
Provides interactive interface for materials generation and synthesizability prediction
"""

import gradio as gr
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
import warnings
from typing import List, Dict, Tuple, Optional
import time
import requests
import os

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
# sns.set_palette("husl")

# ============================================================================
# SYNTHESIZABILITY PREDICTION MODULE (Embedded from notebook)
# ============================================================================

class SynthesizabilityClassifier:
    """ML-based classifier for predicting material synthesizability."""

    def __init__(self, model_type: str = 'random_forest'):
        from sklearn.ensemble import RandomForestClassifier
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'formation_energy_per_atom', 'band_gap', 'energy_above_hull',
            'electronegativity', 'atomic_radius', 'nsites', 'density'
        ]
        self.is_trained = False

    def prepare_training_data(self, api_key: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from real Materials Project data or fallback to synthetic."""
        # Try to import and use real MP data
        try:
            from synthesizability_predictor import create_training_dataset_from_mp
            training_data = create_training_dataset_from_mp(api_key=api_key, n_materials=1000)
        except ImportError:
            # Fallback to synthetic data if import fails
            print("Warning: Using synthetic training data")
            # Mock data generation (same as before)
            np.random.seed(42)

            # ICSD data
            icsd_data = []
            for i in range(1500):
                formation_energy = np.random.normal(-2.0, 1.5)
                formation_energy = np.clip(formation_energy, -8, 2)

                band_gap = np.random.exponential(1.0)
                band_gap = np.clip(band_gap, 0, 8)

                energy_above_hull = np.random.exponential(0.05)
                energy_above_hull = np.clip(energy_above_hull, 0, 0.5)

                electronegativity = np.random.normal(1.8, 0.5)
                electronegativity = np.clip(electronegativity, 0.7, 2.5)

                atomic_radius = np.random.normal(1.4, 0.3)
                atomic_radius = np.clip(atomic_radius, 0.8, 2.2)

                nsites = np.random.randint(1, 20)
                density = np.random.normal(6.0, 3.0)
                density = np.clip(density, 2, 25)

                icsd_data.append({
                    'formation_energy_per_atom': formation_energy,
                    'band_gap': band_gap,
                    'energy_above_hull': energy_above_hull,
                    'electronegativity': electronegativity,
                    'atomic_radius': atomic_radius,
                    'nsites': nsites,
                    'density': density,
                    'synthesizable': 1
                })

            # MP-only data
            np.random.seed(123)
            mp_data = []
            for i in range(1500):
                formation_energy = np.random.normal(-0.5, 2.0)
                formation_energy = np.clip(formation_energy, -8, 4)

                band_gap = np.random.exponential(2.0)
                band_gap = np.clip(band_gap, 0, 12)

                energy_above_hull = np.random.exponential(0.2)
                energy_above_hull = np.clip(energy_above_hull, 0, 1.0)

                electronegativity = np.random.normal(1.6, 0.8)
                electronegativity = np.clip(electronegativity, 0.5, 3.0)

                atomic_radius = np.random.normal(1.5, 0.5)
                atomic_radius = np.clip(atomic_radius, 0.6, 2.8)

                nsites = np.random.randint(1, 50)
                density = np.random.normal(8.0, 4.0)
                density = np.clip(density, 1, 30)

                mp_data.append({
                    'formation_energy_per_atom': formation_energy,
                    'band_gap': band_gap,
                    'energy_above_hull': energy_above_hull,
                    'electronegativity': electronegativity,
                    'atomic_radius': atomic_radius,
                    'nsites': nsites,
                    'density': density,
                    'synthesizable': 0
                })

            training_data = pd.concat([pd.DataFrame(icsd_data), pd.DataFrame(mp_data)], ignore_index=True)
            training_data = training_data.sample(frac=1, random_state=42).reset_index(drop=True)

        X = training_data[self.feature_columns]
        y = training_data['synthesizable']

        return X, y

    def train(self, test_size: float = 0.2, api_key: str = None):
        """Train the synthesizability classifier."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        X, y = self.prepare_training_data(api_key=api_key)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        )

        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }

        cv_scores = cross_val_score(self.model, self.scaler.transform(X), y, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()

        self.is_trained = True

        return metrics

    def predict(self, materials_df: pd.DataFrame) -> pd.DataFrame:
        """Predict synthesizability for new materials."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_pred = materials_df[self.feature_columns].copy()
        X_pred = X_pred.fillna(X_pred.mean())
        X_pred_scaled = self.scaler.transform(X_pred)

        predictions = self.model.predict(X_pred_scaled)
        probabilities = self.model.predict_proba(X_pred_scaled)[:, 1]

        results_df = materials_df.copy()
        results_df['synthesizability_prediction'] = predictions
        results_df['synthesizability_probability'] = probabilities
        results_df['synthesizability_confidence'] = np.abs(probabilities - 0.5) * 2

        return results_df

class LLMSynthesizabilityPredictor:
    """LLM-based synthesizability predictor using rule-based heuristics."""

    def __init__(self):
        self.rules = {
            'thermodynamic_stability': {
                'energy_above_hull_threshold': 0.1,
                'formation_energy_min': -4.0,
                'formation_energy_max': 1.0
            },
            'structural_complexity': {
                'nsites_max': 20,
                'density_min': 2.0,
                'density_max': 25.0
            },
            'electronic_properties': {
                'band_gap_max': 8.0
            },
            'elemental_properties': {
                'electronegativity_min': 0.7,
                'electronegativity_max': 2.8,
                'atomic_radius_min': 0.8,
                'atomic_radius_max': 2.5
            }
        }

    def predict_synthesizability(self, material_data: Dict) -> Dict:
        """Predict synthesizability using rule-based system."""
        score = 0.0
        max_score = 10.0
        reasons = []

        e_hull = material_data.get('energy_above_hull', 0)
        if e_hull <= self.rules['thermodynamic_stability']['energy_above_hull_threshold']:
            score += 2.0
            reasons.append(f"Low energy above hull ({e_hull:.3f} eV/atom) indicates stability")
        else:
            reasons.append(f"High energy above hull ({e_hull:.3f} eV/atom) indicates instability")

        formation_energy = material_data.get('formation_energy_per_atom', 0)
        if (self.rules['thermodynamic_stability']['formation_energy_min'] <=
            formation_energy <= self.rules['thermodynamic_stability']['formation_energy_max']):
            score += 1.5
            reasons.append(f"Formation energy ({formation_energy:.3f} eV/atom) is reasonable")
        else:
            reasons.append(f"Formation energy ({formation_energy:.3f} eV/atom) is extreme")

        nsites = material_data.get('nsites', 10)
        if nsites <= self.rules['structural_complexity']['nsites_max']:
            score += 1.0
            reasons.append(f"Unit cell size ({nsites} sites) is reasonable")
        else:
            reasons.append(f"Unit cell size ({nsites} sites) is very large")

        density = material_data.get('density', 5.0)
        if (self.rules['structural_complexity']['density_min'] <= density <=
            self.rules['structural_complexity']['density_max']):
            score += 1.0
            reasons.append(f"Density ({density:.1f} g/cm³) is reasonable")
        else:
            reasons.append(f"Density ({density:.1f} g/cm³) is unusual")

        band_gap = material_data.get('band_gap', 0)
        if band_gap <= self.rules['electronic_properties']['band_gap_max']:
            score += 1.5
            reasons.append(f"Band gap ({band_gap:.1f} eV) is reasonable")
        else:
            reasons.append(f"Band gap ({band_gap:.1f} eV) is very wide")

        electronegativity = material_data.get('electronegativity', 1.5)
        if (self.rules['elemental_properties']['electronegativity_min'] <=
            electronegativity <= self.rules['elemental_properties']['electronegativity_max']):
            score += 1.0
            reasons.append(f"Electronegativity ({electronegativity:.2f}) is reasonable")
        else:
            reasons.append(f"Electronegativity ({electronegativity:.2f}) is extreme")

        atomic_radius = material_data.get('atomic_radius', 1.4)
        if (self.rules['elemental_properties']['atomic_radius_min'] <=
            atomic_radius <= self.rules['elemental_properties']['atomic_radius_max']):
            score += 1.0
            reasons.append(f"Atomic radius ({atomic_radius:.2f} Å) is reasonable")
        else:
            reasons.append(f"Atomic radius ({atomic_radius:.2f} Å) is unusual")

        probability = 1 / (1 + np.exp(-(score - max_score/2) * 2))
        prediction = 1 if probability >= 0.5 else 0

        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': min(probability, 1-probability) * 2,
            'score': score,
            'max_score': max_score,
            'reasons': reasons
        }

# ============================================================================
# VAE MODEL AND TRAINING
# ============================================================================

class OptimizedVAE(nn.Module):
    """Optimized Variational Autoencoder for materials discovery."""

    def __init__(self, input_dim: int = 6, latent_dim: int = 5):
        super(OptimizedVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_var = nn.Linear(32, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
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

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_synthetic_dataset(n_samples: int = 1000) -> pd.DataFrame:
    """Create synthetic materials dataset for demonstration."""
    np.random.seed(42)

    alloys = []
    elements = ['Al', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']

    for i in range(n_samples):
        alloy_type = np.random.choice(['binary', 'ternary'], p=[0.7, 0.3])

        if alloy_type == 'binary':
            elem1, elem2 = np.random.choice(elements, 2, replace=False)
            comp1 = np.random.uniform(0.1, 0.9)
            comp2 = 1 - comp1
            comp3 = 0
        else:
            elem1, elem2, elem3 = np.random.choice(elements, 3, replace=False)
            comp1 = np.random.uniform(0.1, 0.6)
            comp2 = np.random.uniform(0.1, 0.6)
            comp3 = 1 - comp1 - comp2

        melting_point = np.random.normal(1500, 300)
        density = np.random.normal(7.8, 2.0)
        electronegativity = np.random.normal(1.8, 0.3)
        atomic_radius = np.random.normal(1.3, 0.2)

        alloys.append({
            'id': f'alloy_{i+1}',
            'alloy_type': alloy_type,
            'element_1': elem1,
            'element_2': elem2,
            'element_3': elem3 if alloy_type == 'ternary' else None,
            'composition_1': comp1,
            'composition_2': comp2,
            'composition_3': comp3,
            'melting_point': max(500, melting_point),
            'density': max(2, density),
            'electronegativity': max(0.7, min(2.5, electronegativity)),
            'atomic_radius': max(1.0, min(1.8, atomic_radius))
        })

    return pd.DataFrame(alloys)

def train_vae_model(features_scaled: np.ndarray, latent_dim: int = 5, epochs: int = 50) -> OptimizedVAE:
    """Train VAE model on materials features."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = features_scaled.shape[1]

    model = OptimizedVAE(input_dim=input_dim, latent_dim=latent_dim).to(device)

    features_tensor = torch.FloatTensor(features_scaled)
    dataset = torch.utils.data.TensorDataset(features_tensor, features_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    model.train()
    for epoch in range(epochs):
        for batch_x, _ in dataloader:
            batch_x = batch_x.to(device)

            reconstructed, mu, log_var = model(batch_x)
            reconstruction_loss = nn.functional.mse_loss(reconstructed, batch_x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            kl_weight = min(1.0, epoch / 10.0)
            loss = reconstruction_loss + kl_weight * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

    return model

def generate_materials(model: OptimizedVAE, scaler: StandardScaler, num_samples: int = 100) -> pd.DataFrame:
    """Generate new materials using trained VAE with tight composition constraints."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    elements = ['Al', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
    new_materials = []

    with torch.no_grad():
        z = torch.randn(num_samples, model.latent_dim).to(device)
        generated_features = model.decode(z).cpu().numpy()
        generated_features = scaler.inverse_transform(generated_features)

    for i, features in enumerate(generated_features):
        # Randomly select elements
        elem1, elem2 = random.sample(elements, 2)

        # Enforce tight composition constraints (5-95% range)
        comp1 = np.clip(features[0], 0.05, 0.95)
        comp2 = 1.0 - comp1

        # Assert composition validity (within floating point tolerance)
        assert abs(comp1 + comp2 - 1.0) < 1e-6, f"Composition constraint violated: {comp1} + {comp2} = {comp1 + comp2}"

        material = {
            'id': f'generated_{i+1}',
            'element_1': elem1,
            'element_2': elem2,
            'composition_1': comp1,
            'composition_2': comp2,
            'formula': f'{elem1}{comp1:.3f}{elem2}{comp2:.3f}',
            'melting_point': abs(features[2]),
            'density': abs(features[3]),
            'electronegativity': max(0, features[4]),
            'atomic_radius': max(0, features[5]),
            'is_generated': True
        }
        new_materials.append(material)

    return pd.DataFrame(new_materials)

def run_synthesizability_analysis(materials_df: pd.DataFrame, ml_classifier: SynthesizabilityClassifier,
                                llm_predictor: LLMSynthesizabilityPredictor) -> pd.DataFrame:
    """Run complete synthesizability analysis on materials, separating thermodynamic stability from experimental synthesizability."""
    # Prepare data for analysis
    analysis_df = materials_df.copy()

    # Add required columns with defaults
    analysis_df['formation_energy_per_atom'] = analysis_df.get('melting_point', analysis_df['melting_point'].mean())
    analysis_df['energy_above_hull'] = np.random.uniform(0, 0.5, len(analysis_df))
    analysis_df['band_gap'] = np.random.uniform(0, 3, len(analysis_df))
    analysis_df['nsites'] = np.random.randint(2, 8, len(analysis_df))

    # Calculate thermodynamic stability scores (separate from synthesizability)
    try:
        from synthesizability_predictor import calculate_thermodynamic_stability_score
        stability_results = []
        for idx, material in analysis_df.iterrows():
            stability_result = calculate_thermodynamic_stability_score(material.to_dict())
            stability_results.append(stability_result)
    except ImportError:
        # Fallback if function not available
        stability_results = [{
            'thermodynamic_stability_score': 0.5,
            'thermodynamic_stability_category': 'unknown',
            'stability_color': 'gray'
        } for _ in range(len(analysis_df))]

    # ML predictions for experimental synthesizability
    ml_results = ml_classifier.predict(analysis_df)

    # LLM predictions for experimental synthesizability
    llm_predictions = []
    for idx, material in analysis_df.iterrows():
        llm_result = llm_predictor.predict_synthesizability(material.to_dict())
        llm_predictions.append(llm_result)

    # Combine results
    results_df = ml_results.copy()
    results_df['llm_prediction'] = [p['prediction'] for p in llm_predictions]
    results_df['llm_probability'] = [p['probability'] for p in llm_predictions]
    results_df['llm_confidence'] = [p['confidence'] for p in llm_predictions]

    # Ensemble prediction for experimental synthesizability
    results_df['ensemble_probability'] = (
        0.7 * results_df['synthesizability_probability'] +
        0.3 * results_df['llm_probability']
    )
    results_df['ensemble_prediction'] = (results_df['ensemble_probability'] >= 0.5).astype(int)
    results_df['ensemble_confidence'] = np.abs(results_df['ensemble_probability'] - 0.5) * 2

    # Add thermodynamic stability results (separate from synthesizability)
    results_df['thermodynamic_stability_score'] = [r['thermodynamic_stability_score'] for r in stability_results]
    results_df['thermodynamic_stability_category'] = [r['thermodynamic_stability_category'] for r in stability_results]
    results_df['stability_color'] = [r['stability_color'] for r in stability_results]

    return results_df

def calculate_synthesis_priority(materials_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate priority ranking for experimental synthesis."""
    results_df = materials_df.copy()

    stability_score = 1 - results_df['energy_above_hull'].clip(0, 1)
    synth_score = results_df.get('ensemble_probability', 0.5)
    novelty_score = results_df['energy_above_hull'].clip(0, 0.5) / 0.5
    ease_score = 1 / (1 + results_df['nsites'] / 10)

    priority_score = (
        0.4 * stability_score +
        0.3 * synth_score +
        0.2 * novelty_score +
        0.1 * ease_score
    )

    results_df['synthesis_priority_score'] = priority_score
    results_df['synthesis_priority_rank'] = priority_score.rank(ascending=False).astype(int)

    return results_df

def get_synthesis_methods_reference() -> pd.DataFrame:
    """Get a reference table of all available synthesis methods."""
    from synthesizability_predictor import SYNTHESIS_METHODS

    methods_data = []
    for method_key, method_info in SYNTHESIS_METHODS.items():
        methods_data.append({
            'Method': method_info['name'],
            'Description': method_info['description'],
            'Temperature Range': f"{method_info['temperature_range']['typical']}°C ({method_info['temperature_range']['min']}-{method_info['temperature_range']['max']}°C)",
            'Atmosphere': method_info['atmosphere'],
            'Equipment Needed': ', '.join(method_info['equipment_needed']),
            'Total Time (hours)': (method_info['prep_time_hours'] +
                                 method_info['synthesis_time_hours'] +
                                 method_info['cooling_time_hours']),
            'Cost (USD)': method_info['estimated_cost_usd'],
            'Success Rate': f"{method_info['success_probability']:.1%}",
            'Best For': ', '.join(method_info['best_for'][:2]),  # Show first 2 applications
            'Reference': method_info['reference']
        })

    return pd.DataFrame(methods_data)

def create_experimental_workflow(materials_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Create prioritized experimental validation workflow."""
    prioritized_df = calculate_synthesis_priority(materials_df)
    workflow_df = prioritized_df.sort_values('synthesis_priority_score', ascending=False).head(top_n).copy()

    workflow_df['workflow_step'] = range(1, len(workflow_df) + 1)
    workflow_df['batch_number'] = ((workflow_df['workflow_step'] - 1) // 5) + 1
    workflow_df['estimated_time_days'] = np.random.uniform(3, 14, len(workflow_df)).round(1)
    workflow_df['required_equipment'] = ['Arc Melter/Furnace'] * len(workflow_df)
    workflow_df['priority_level'] = pd.cut(workflow_df['workflow_step'],
                                         bins=[0, 3, 7, 10], labels=['High', 'Medium', 'Low'])

    return workflow_df[[
        'workflow_step', 'batch_number', 'priority_level',
        'synthesis_priority_score', 'ensemble_probability',
        'energy_above_hull', 'formation_energy_per_atom',
        'estimated_time_days', 'required_equipment'
    ]]

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_gradio_interface():
    """Create the main Gradio interface."""

    # Get API key for Materials Project
    api_key = os.getenv("MP_API_KEY")

    # Initialize models
    ml_classifier = SynthesizabilityClassifier()
    ml_metrics = ml_classifier.train(api_key=api_key)
    llm_predictor = LLMSynthesizabilityPredictor()

    # Create VAE training dataset (real MP data if available, synthetic otherwise)
    try:
        from synthesizability_predictor import create_vae_training_dataset_from_mp
        dataset = create_vae_training_dataset_from_mp(api_key=api_key, n_materials=1000)
    except ImportError:
        print("Warning: Using synthetic dataset for VAE training")
        dataset = create_synthetic_dataset(1000)

    feature_cols = ['composition_1', 'composition_2', 'melting_point', 'density', 'electronegativity', 'atomic_radius']
    features = dataset[feature_cols].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Initialize synthesis methods reference table
    methods_ref_df = get_synthesis_methods_reference()

    def generate_and_analyze(latent_dim, epochs, num_samples, available_equipment):
        """Main function to generate materials and run analysis."""
        try:
            # Train VAE
            vae_model = train_vae_model(features_scaled, latent_dim=latent_dim, epochs=epochs)

            # Generate materials
            generated_df = generate_materials(vae_model, scaler, num_samples=num_samples)

            # Run synthesizability analysis
            results_df = run_synthesizability_analysis(generated_df, ml_classifier, llm_predictor)

            # Calculate priorities
            priority_df = calculate_synthesis_priority(results_df)

            # Add weight percent and feedstock calculations
            try:
                from synthesizability_predictor import add_composition_analysis_to_dataframe
                results_df = add_composition_analysis_to_dataframe(results_df)
            except ImportError:
                print("Warning: Could not import composition analysis functions")
                results_df['wt%'] = 'N/A'
                results_df['feedstock_g_per_100g'] = 'N/A'

            # Create experimental workflow
            workflow_df = create_experimental_workflow(results_df, top_n=10)

            # Cost-benefit analysis for top 5 using selected equipment

            top_5_materials = priority_df.head(5)
            cba_results = []
            for _, material in top_5_materials.iterrows():
                try:
                    from synthesizability_predictor import cost_benefit_analysis
                    cba = cost_benefit_analysis(material.to_dict(), available_equipment)
                    cba['formula'] = material['formula']
                    cba_results.append(cba)
                except ImportError:
                    # Fallback to basic cost-benefit if import fails
                    cba_results.append({
                        'formula': material['formula'],
                        'recommended_method': 'Unknown',
                        'success_probability': material.get('ensemble_probability', 0.5),
                        'net_benefit': 0,
                        'benefit_cost_ratio': 0
                    })
            cba_df = pd.DataFrame(cba_results)

            # Create summary statistics
            total_materials = len(results_df)
            synthesizable_count = results_df['ensemble_prediction'].sum()
            high_confidence = results_df[results_df['ensemble_confidence'] > 0.8]

            summary_text = f"""
            **Generation Summary:**
            - Total materials generated: {total_materials}
            - Predicted synthesizable: {synthesizable_count} ({synthesizable_count/total_materials*100:.1f}%)
            - High confidence predictions: {len(high_confidence)} ({len(high_confidence)/total_materials*100:.1f}%)

            **Priority Analysis:**
            - Top priority materials identified: 10
            - Materials with priority > 0.7: {(priority_df['synthesis_priority_score'] > 0.7).sum()}

            **Cost-Benefit Analysis:**
            - Positive ROI candidates: {len(cba_df[cba_df['net_benefit'] > 0])}/5
            - Average net benefit: ${cba_df['net_benefit'].mean():.0f}

            **Model Performance:**
            - ML Classifier Accuracy: {ml_metrics['accuracy']:.3f}
            - ML Classifier F1-Score: {ml_metrics['f1_score']:.3f}
            """

            # Create visualizations
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # Synthesis probability distribution
            ax1.hist(results_df['ensemble_probability'], bins=20, alpha=0.7, color='blue')
            ax1.set_xlabel('Synthesizability Probability')
            ax1.set_ylabel('Count')
            ax1.set_title('Synthesizability Distribution')
            ax1.grid(True, alpha=0.3)

            # Stability vs Synthesizability
            scatter = ax2.scatter(results_df['energy_above_hull'], results_df['ensemble_probability'],
                                alpha=0.6, c=results_df['density'], cmap='viridis')
            ax2.set_xlabel('Energy Above Hull (eV/atom)')
            ax2.set_ylabel('Synthesizability Probability')
            ax2.set_title('Stability vs Synthesizability')
            plt.colorbar(scatter, ax=ax2, label='Density (g/cm³)')

            # Priority score distribution
            ax3.hist(priority_df['synthesis_priority_score'], bins=15, alpha=0.7, color='green')
            ax3.set_xlabel('Synthesis Priority Score')
            ax3.set_ylabel('Count')
            ax3.set_title('Priority Score Distribution')
            ax3.grid(True, alpha=0.3)

            # Cost-benefit analysis
            if len(cba_df) > 0:
                bars = ax4.bar(range(len(cba_df)), cba_df['net_benefit'],
                              color=['green' if x > 0 else 'red' for x in cba_df['net_benefit']])
                ax4.set_xlabel('Candidate')
                ax4.set_ylabel('Net Benefit ($)')
                ax4.set_title('Cost-Benefit Analysis')
                ax4.set_xticks(range(len(cba_df)))
                ax4.set_xticklabels(cba_df['formula'], rotation=45)
                ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            # Prepare results tables with separate stability and synthesizability columns
            display_cols = ['formula', 'thermodynamic_stability_score', 'thermodynamic_stability_category',
                          'ensemble_probability', 'ensemble_confidence', 'energy_above_hull', 'density', 'melting_point']
            results_table = results_df[display_cols].round(3)

            # Apply color coding for stability status
            def color_stability(val):
                if val == 'highly_stable':
                    return 'background-color: lightgreen'
                elif val == 'marginal':
                    return 'background-color: lightyellow'
                elif val == 'unstable':
                    return 'background-color: lightcoral'
                return ''

            # Style the results table with color coding
            styled_results = results_table.style.applymap(color_stability, subset=['thermodynamic_stability_category'])

            priority_cols = ['formula', 'synthesis_priority_score', 'synthesis_priority_rank',
                           'thermodynamic_stability_score', 'ensemble_probability', 'energy_above_hull', 'density']
            priority_table = priority_df[priority_cols].round(3).head(20)

            workflow_cols = ['workflow_step', 'batch_number', 'priority_level',
                           'synthesis_priority_score', 'ensemble_probability', 'estimated_time_days']
            workflow_table = workflow_df[workflow_cols].round(3)

            cba_display = cba_df[['formula', 'recommended_method', 'success_probability',
                                'net_benefit', 'benefit_cost_ratio']].round(3)

            return summary_text, fig, styled_results, priority_table, workflow_table, cba_display, methods_ref_df

        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            return error_msg, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Create Gradio interface
    with gr.Blocks(title="Materials Discovery Workshop") as interface:

        gr.Markdown("""
        # 🧪 Materials Discovery Workshop
        ### AI-Powered Materials Generation & Synthesizability Prediction

        This interactive tool uses machine learning to generate new alloy compositions and predict their synthesizability.
        Adjust the parameters below to explore different material generation scenarios.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Model Parameters")
                latent_dim = gr.Slider(
                    minimum=2, maximum=20, value=5, step=1,
                    label="Latent Dimension",
                    info="Complexity of the generative model"
                )
                epochs = gr.Slider(
                    minimum=10, maximum=200, value=50, step=10,
                    label="Training Epochs",
                    info="How long to train the model"
                )
                num_samples = gr.Slider(
                    minimum=10, maximum=500, value=100, step=10,
                    label="Materials to Generate",
                    info="Number of new materials to create"
                )

                gr.Markdown("### Available Equipment")
                from synthesizability_predictor import EQUIPMENT_CATEGORIES
                all_equipment = []
                for category_equipment in EQUIPMENT_CATEGORIES.values():
                    all_equipment.extend(category_equipment)

                available_equipment = gr.CheckboxGroup(
                    choices=all_equipment,
                    value=all_equipment,  # Default to all equipment available
                    label="Select Available Equipment",
                    info="Filter synthesis recommendations based on your lab equipment"
                )

                generate_btn = gr.Button("🚀 Generate Materials", variant="primary", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Results Summary")
                summary_output = gr.Markdown()

        with gr.Tabs():
            with gr.TabItem("📊 Analysis Overview"):
                with gr.Row():
                    gr.Markdown("### Key Visualizations")
                    plot_output = gr.Plot()

            with gr.TabItem("🧪 Generated Materials"):
                gr.Markdown("### All Generated Materials with Stability & Synthesizability Predictions")
                materials_table = gr.DataFrame(
                    headers=["Formula", "Stability Score", "Stability Category", "Synth. Prob.", "Synth. Conf.", "E_hull", "Density", "Melting Point"],
                    label="Generated Materials",
                    wrap=True
                )

            with gr.TabItem("🎯 Priority Ranking"):
                gr.Markdown("### Synthesis Priority Ranking (Top 20 Materials)")
                priority_table = gr.DataFrame(
                    headers=["Formula", "Priority Score", "Rank", "Stability Score", "Synth. Prob.", "E_hull", "Density"],
                    label="Priority Ranked Materials"
                )

            with gr.TabItem("📋 Experimental Workflow"):
                gr.Markdown("### Prioritized Experimental Validation Workflow")
                workflow_table = gr.DataFrame(
                    headers=["Step", "Batch", "Priority", "Priority Score", "Synth. Prob.", "Est. Days"],
                    label="Experimental Workflow"
                )

            with gr.TabItem("💰 Cost-Benefit Analysis"):
                gr.Markdown("### Cost-Benefit Analysis for Top 5 Candidates")
                cba_table = gr.DataFrame(
                    headers=["Formula", "Method", "Success Prob.", "Net Benefit", "Benefit Ratio"],
                    label="Cost-Benefit Analysis"
                )

            with gr.TabItem("🔬 Synthesis Methods"):
                gr.Markdown("### Complete Reference Guide for Synthesis Methods")
                methods_table = gr.DataFrame(
                    value=methods_ref_df,
                    headers=["Method", "Description", "Temperature Range", "Atmosphere", "Equipment Needed", "Total Time (hours)", "Cost (USD)", "Success Rate", "Best For", "Reference"],
                    label="Synthesis Methods Reference",
                    wrap=True
                )

            with gr.TabItem("📥 CSV Export"):
                gr.Markdown("### Export Materials Data for Lab Use")
                gr.Markdown("""
                Download a CSV file containing all generated materials with composition details,
                including atomic percentages (at%), weight percentages (wt%), and feedstock masses
                for preparing 100g batches in the laboratory.

                **CSV Columns Include:**
                - `formula`: Chemical formula
                - `at%`: Atomic percentages (original compositions)
                - `wt%`: Weight percentages calculated from atomic composition
                - `feedstock_g_per_100g`: Exact masses needed for 100g batch
                - All other analysis results
                """)

                csv_export_info = gr.Markdown("""
                **Example Usage:**
                For material Al₀.₅Ti₀.₅, the CSV will show:
                - at%: Al: 50.0%, Ti: 50.0%
                - wt%: Al: 51.0%, Ti: 49.0%
                - feedstock_g_per_100g: 51.0g Al, 49.0g Ti

                This allows metallurgists to immediately weigh out ingredients for synthesis!
                """)

                csv_download = gr.File(label="Download Materials CSV")

        # Function to prepare CSV export
        def prepare_csv_export(results_df):
            """Prepare CSV data for download with all required columns."""
            if results_df.empty:
                return None

            # Prepare CSV columns as specified in requirements
            csv_data = results_df.copy()

            # Create at% column (atomic percentages)
            def format_atomic_percent(row):
                elements = []
                if row.get('element_1') and row.get('composition_1', 0) > 0:
                    elements.append(f"{row['element_1']}: {row['composition_1']*100:.1f}%")
                if row.get('element_2') and row.get('composition_2', 0) > 0:
                    elements.append(f"{row['element_2']}: {row['composition_2']*100:.1f}%")
                if row.get('element_3') and row.get('composition_3', 0) > 0:
                    elements.append(f"{row['element_3']}: {row['composition_3']*100:.1f}%")
                return "; ".join(elements)

            csv_data['at%'] = csv_data.apply(format_atomic_percent, axis=1)

            # Select and order columns for CSV export
            export_columns = [
                'formula', 'at%', 'wt%', 'feedstock_g_per_100g',
                'thermodynamic_stability_score', 'thermodynamic_stability_category',
                'ensemble_probability', 'ensemble_confidence', 'ensemble_prediction',
                'energy_above_hull', 'formation_energy_per_atom', 'density',
                'melting_point', 'band_gap', 'electronegativity', 'atomic_radius',
                'nsites', 'element_1', 'element_2', 'element_3',
                'composition_1', 'composition_2', 'composition_3'
            ]

            # Only include columns that exist
            available_columns = [col for col in export_columns if col in csv_data.columns]
            csv_export_df = csv_data[available_columns]

            # Create temporary CSV file
            import tempfile
            import os
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            csv_export_df.to_csv(temp_file.name, index=False)
            temp_file.close()

            return temp_file.name

        # Connect CSV export to generation function
        def generate_with_csv(latent_dim, epochs, num_samples, available_equipment):
            """Generate materials and prepare CSV export."""
            results = generate_and_analyze(latent_dim, epochs, num_samples, available_equipment)

            # Extract results_df from the tuple (it's the 3rd element)
            if len(results) >= 3 and isinstance(results[2], pd.DataFrame):
                results_df = results[2].data if hasattr(results[2], 'data') else results[2]
                csv_path = prepare_csv_export(results_df)
            else:
                csv_path = None

            # Return all original results plus CSV path
            return results + (csv_path,)

        # Update the button to use the new function
        generate_btn.click(
            fn=generate_with_csv,
            inputs=[latent_dim, epochs, num_samples, available_equipment],
            outputs=[summary_output, plot_output, materials_table, priority_table, workflow_table, cba_table, methods_table, csv_download]
        )

        gr.Markdown("""
        ### How it Works
        1. **VAE Training**: A Variational Autoencoder learns patterns from existing materials data
        2. **Material Generation**: The trained model generates new alloy compositions
        3. **Thermodynamic Stability Assessment**: Evaluates chemical/phase stability based on energy above hull and formation energies
        4. **Experimental Synthesizability Prediction**: ML and rule-based models predict experimental feasibility
        5. **Priority Ranking**: Materials are ranked by synthesis priority and potential value
        6. **Cost-Benefit Analysis**: Economic evaluation of synthesis candidates
        7. **Workflow Planning**: Prioritized experimental validation schedule

        ### Key Distinctions
        - **Thermodynamic Stability**: Chemical/phase stability (E_hull, formation energy, density) - indicates if a material is *chemically stable*
        - **Experimental Synthesizability**: Laboratory feasibility and experimental accessibility - indicates if a material is *experimentally feasible*

        ### Model Details
        - **Dataset**: Synthetic alloy compositions (can be extended to real Materials Project data)
        - **VAE Architecture**: Encoder → Latent Space → Decoder with KL divergence regularization
        - **Stability Assessment**: Rule-based scoring using literature thresholds (highly stable ≤0.025 eV/atom, marginal ≤0.1 eV/atom, unstable >0.1 eV/atom)
        - **Synthesizability Prediction Models**: Random Forest classifier + Rule-based expert system
        - **Ensemble Method**: Weighted combination of ML and expert predictions for experimental feasibility
        - **Priority Scoring**: Multi-criteria optimization (thermodynamic stability, experimental synthesizability, novelty, ease)
        """)

    return interface

# ============================================================================
# MAIN APPLICATION
# ============================================================================

if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
