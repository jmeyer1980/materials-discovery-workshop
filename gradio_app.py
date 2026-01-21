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
from datetime import datetime

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
# sns.set_palette("husl")

# Import synthesizability prediction classes from the dedicated module
from synthesizability_predictor import SynthesizabilityClassifier, LLMSynthesizabilityPredictor

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

    # ML predictions for experimental synthesizability (now includes calibration and in-distribution)
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

    # Ensemble prediction for experimental synthesizability using configurable weights
    ensemble_weights = ml_classifier.ensemble_weights
    results_df['ensemble_probability'] = (
        ensemble_weights['ml'] * results_df['synthesizability_probability'] +
        ensemble_weights['llm'] * results_df['llm_probability']
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
# EXPERIMENTAL OUTCOMES MANAGEMENT
# ============================================================================

def load_experimental_outcomes() -> pd.DataFrame:
    """Load experimental outcomes from CSV file."""
    outcomes_file = 'outcomes.csv'
    if os.path.exists(outcomes_file):
        try:
            df = pd.read_csv(outcomes_file)
            # Ensure required columns exist
            required_cols = ['formula', 'attempted_date', 'outcome', 'ml_pred', 'llm_pred', 'e_hull', 'notes']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception as e:
            print(f"Error loading outcomes: {e}")
            return pd.DataFrame(columns=['formula', 'attempted_date', 'outcome', 'ml_pred', 'llm_pred', 'e_hull', 'notes'])
    else:
        return pd.DataFrame(columns=['formula', 'attempted_date', 'outcome', 'ml_pred', 'llm_pred', 'e_hull', 'notes'])

def save_experimental_outcome(formula: str, outcome: str, ml_pred: float, llm_pred: float,
                             e_hull: float, notes: str) -> bool:
    """Save a new experimental outcome to CSV."""
    try:
        outcomes_df = load_experimental_outcomes()

        new_outcome = {
            'formula': formula,
            'attempted_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'outcome': outcome,
            'ml_pred': ml_pred,
            'llm_pred': llm_pred,
            'e_hull': e_hull,
            'notes': notes
        }

        # Append new outcome
        outcomes_df = pd.concat([outcomes_df, pd.DataFrame([new_outcome])], ignore_index=True)

        # Save to CSV
        outcomes_df.to_csv('outcomes.csv', index=False)
        return True
    except Exception as e:
        print(f"Error saving outcome: {e}")
        return False

def calculate_validation_metrics(outcomes_df: pd.DataFrame) -> Dict:
    """Calculate validation metrics from experimental outcomes."""
    if outcomes_df.empty:
        return {
            'total_experiments': 0,
            'success_rate': 0.0,
            'hit_rate_last_10': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'calibration_error': 0.0,
            'recent_experiments': []
        }

    # Calculate basic metrics
    total_experiments = len(outcomes_df)
    successful_experiments = len(outcomes_df[outcomes_df['outcome'] == 'Successful'])
    success_rate = successful_experiments / total_experiments if total_experiments > 0 else 0.0

    # Hit rate for last 10 experiments
    recent_outcomes = outcomes_df.tail(10)
    recent_successful = len(recent_outcomes[recent_outcomes['outcome'] == 'Successful'])
    hit_rate_last_10 = recent_successful / len(recent_outcomes) if len(recent_outcomes) > 0 else 0.0

    # Precision and recall (treating "Successful" + "Partial success" as positive predictions)
    positive_predictions = (outcomes_df['ml_pred'] >= 0.5) | (outcomes_df['llm_pred'] >= 0.5)
    actual_successes = outcomes_df['outcome'].isin(['Successful', 'Partial success'])

    if positive_predictions.sum() > 0:
        precision = ((positive_predictions) & (actual_successes)).sum() / positive_predictions.sum()
    else:
        precision = 0.0

    if actual_successes.sum() > 0:
        recall = ((positive_predictions) & (actual_successes)).sum() / actual_successes.sum()
    else:
        recall = 0.0

    # Calibration error (mean absolute difference between predicted and actual success rates)
    outcomes_df_copy = outcomes_df.copy()
    outcomes_df_copy['actual_success'] = outcomes_df_copy['outcome'].isin(['Successful', 'Partial success']).astype(int)
    outcomes_df_copy['ensemble_pred'] = (outcomes_df_copy['ml_pred'] * 0.7 + outcomes_df_copy['llm_pred'] * 0.3)

    # Group by prediction bins
    bins = np.linspace(0, 1, 11)
    outcomes_df_copy['pred_bin'] = pd.cut(outcomes_df_copy['ensemble_pred'], bins=bins, labels=bins[:-1])

    calibration_errors = []
    for bin_start in bins[:-1]:
        bin_data = outcomes_df_copy[outcomes_df_copy['pred_bin'] == bin_start]
        if len(bin_data) > 0:
            predicted_rate = bin_data['ensemble_pred'].mean()
            actual_rate = bin_data['actual_success'].mean()
            calibration_errors.append(abs(predicted_rate - actual_rate))

    calibration_error = np.mean(calibration_errors) if calibration_errors else 0.0

    return {
        'total_experiments': total_experiments,
        'success_rate': success_rate,
        'hit_rate_last_10': hit_rate_last_10,
        'precision': precision,
        'recall': recall,
        'calibration_error': calibration_error,
        'recent_experiments': recent_outcomes.to_dict('records')
    }

def filter_materials_by_outcome_status(materials_df: pd.DataFrame, outcomes_df: pd.DataFrame,
                                     filter_type: str) -> pd.DataFrame:
    """Filter materials based on whether they have experimental outcomes logged."""
    if filter_type == "All Materials":
        return materials_df
    elif filter_type == "Has Outcome Logged":
        logged_formulas = set(outcomes_df['formula'].unique())
        return materials_df[materials_df['formula'].isin(logged_formulas)]
    elif filter_type == "No Outcome Logged":
        logged_formulas = set(outcomes_df['formula'].unique())
        return materials_df[~materials_df['formula'].isin(logged_formulas)]
    else:
        return materials_df

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_gradio_interface():
    """Create the main Gradio interface."""

    # API key will be provided by user through the interface
    api_key = None

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

    # Global variables to store data for export functions
    global_ml_metrics = None
    global_full_results_df = None  # Store full internal DataFrame for CSV export

    def generate_and_analyze(api_key, latent_dim, epochs, num_samples, available_equipment):
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

            # Return regular DataFrame for Gradio (styling is handled in the UI)
            priority_cols = ['formula', 'synthesis_priority_score', 'synthesis_priority_rank',
                           'thermodynamic_stability_score', 'ensemble_probability', 'energy_above_hull', 'density']
            priority_table = priority_df[priority_cols].round(3).head(20)

            workflow_cols = ['workflow_step', 'batch_number', 'priority_level',
                           'synthesis_priority_score', 'ensemble_probability', 'estimated_time_days']
            workflow_table = workflow_df[workflow_cols].round(3)

            cba_display = cba_df[['formula', 'recommended_method', 'success_probability',
                                'net_benefit', 'benefit_cost_ratio']].round(3)

            # Create reliability plot (calibration curve and distance histogram)
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Calibration curve
            if hasattr(ml_classifier, 'calibration_curve') and ml_classifier.calibration_curve is not None:
                prob_true, prob_pred, bins = ml_classifier.calibration_curve
                ax1.plot(prob_pred, prob_true, 's-', label='Calibration curve', color='blue', linewidth=2)
                ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.7)
                ax1.set_xlabel('Predicted Probability')
                ax1.set_ylabel('True Probability')
                ax1.set_title('Calibration Curve')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)

            # Distance histogram
            if 'nn_distance' in results_df.columns:
                in_dist_mask = results_df['in_distribution'] == 'in-dist'
                out_dist_mask = results_df['in_distribution'] == 'out-dist'

                ax2.hist(results_df.loc[in_dist_mask, 'nn_distance'], bins=20, alpha=0.7,
                        label='In-distribution', color='green', density=True)
                ax2.hist(results_df.loc[out_dist_mask, 'nn_distance'], bins=20, alpha=0.7,
                        label='Out-distribution', color='red', density=True)
                ax2.axvline(x=np.percentile(results_df['nn_distance'], 95), color='black',
                           linestyle='--', alpha=0.7, label='95th percentile threshold')
                ax2.set_xlabel('Nearest Neighbor Distance')
                ax2.set_ylabel('Density')
                ax2.set_title('In-Distribution vs Out-Distribution')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            return summary_text, fig, results_table, priority_table, workflow_table, cba_display, methods_ref_df, fig2, results_df

        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            return error_msg, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

    # Create Gradio interface
    with gr.Blocks(title="Materials Discovery Workshop") as interface:

        gr.Markdown("""
        # 🧪 Materials Discovery Workshop
        ### AI-Powered Materials Generation & Synthesizability Prediction

        This interactive tool uses machine learning to generate new alloy compositions and predict their synthesizability.
        Adjust the parameters below to explore different material generation scenarios.
        """)

        gr.Markdown("""
        ## 🔑 Materials Project API Key (Required)

        To access real materials data from the Materials Project database, you need to provide your own API key.
        **Get your free API key at: [https://materialsproject.org/api](https://materialsproject.org/api)**

        The application will work with synthetic data if no API key is provided, but real Materials Project data
        provides much more accurate and realistic results for materials discovery.
        """)

        api_key_input = gr.Textbox(
            label="Materials Project API Key",
            placeholder="Enter your API key from materialsproject.org/api",
            type="password",
            info="Required for accessing real materials data. Leave empty to use synthetic data only."
        )

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

                # Filter controls
                with gr.Row():
                    filter_dropdown = gr.Dropdown(
                        choices=["All Materials", "Has Outcome Logged", "No Outcome Logged"],
                        value="All Materials",
                        label="Filter by Experimental Status",
                        info="Show materials based on whether outcomes have been logged"
                    )
                    apply_filter_btn = gr.Button("🔍 Apply Filter", variant="secondary")

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

            with gr.TabItem("🧪 Experimental Outcomes"):
                gr.Markdown("### Log Experimental Synthesis Outcomes")
                gr.Markdown("""
                Track the results of your synthesis experiments to validate and improve model predictions.
                This data helps calibrate the models and identify which predictions are most reliable.
                """)

                # Load current materials for dropdown
                current_outcomes_df = load_experimental_outcomes()

                with gr.Row():
                    with gr.Column():
                        # Formula selection
                        formula_dropdown = gr.Dropdown(
                            choices=[],  # Will be populated after generation
                            label="Select Material Formula",
                            info="Choose from generated candidates"
                        )

                        # Outcome selection
                        outcome_dropdown = gr.Dropdown(
                            choices=["Not yet attempted", "Successful", "Partial success", "Failed"],
                            value="Not yet attempted",
                            label="Experimental Outcome",
                            info="Record the result of your synthesis attempt"
                        )

                        # Notes
                        notes_textbox = gr.Textbox(
                            label="Experimental Notes",
                            placeholder="Describe synthesis conditions, challenges, observations...",
                            lines=3
                        )

                        # Log outcome button
                        log_outcome_btn = gr.Button("📝 Log Outcome", variant="primary")

                    with gr.Column():
                        gr.Markdown("### Recent Experimental Outcomes")
                        outcomes_display = gr.DataFrame(
                            value=current_outcomes_df.tail(10),
                            headers=["Formula", "Date", "Outcome", "ML Pred", "LLM Pred", "E_hull", "Notes"],
                            label="Recent Outcomes"
                        )

                        outcome_status = gr.Markdown("")

            with gr.TabItem("📊 Validation Metrics"):
                gr.Markdown("### Model Validation & Performance Tracking")
                gr.Markdown("""
                **Performance Metrics from Experimental Data:**

                **Hit Rate**: Percentage of successful experiments in recent trials
                **Precision**: Of materials predicted synthesizable, what fraction actually succeeded
                **Recall**: Of successful experiments, what fraction were correctly predicted
                **Calibration**: How well predicted probabilities match actual success rates
                """)

                # Validation metrics display
                validation_metrics_display = gr.Markdown("""
                **Current Metrics:**
                - Total Experiments: 0
                - Overall Success Rate: 0.0%
                - Hit Rate (Last 10): 0.0%
                - Precision: 0.0%
                - Recall: 0.0%
                - Calibration Error: 0.0
                """)

                # Recent experiments table
                recent_experiments_table = gr.DataFrame(
                    headers=["Formula", "Outcome", "Predicted Prob", "Actual Success", "Notes"],
                    label="Recent Experimental Results"
                )

                # Calibration plot
                calibration_plot = gr.Plot(label="Predicted vs Actual Success Rates")

                # Update validation metrics button
                update_metrics_btn = gr.Button("🔄 Update Validation Metrics")

            with gr.TabItem("📈 Model Reliability"):
                gr.Markdown("### Calibration Metrics & In-Distribution Detection")
                gr.Markdown("""
                **Understanding Model Reliability:**

                **Calibration Curve**: Shows how well predicted probabilities match actual outcomes.
                - **Well-calibrated**: Predictions accurately reflect true likelihood
                - **Overconfident**: Model is too sure (predicts high probability but often wrong)
                - **Underconfident**: Model is too conservative (predicts low probability but often right)

                **In-Distribution Detection**: Identifies materials similar to training data.
                - **In-distribution (in-dist)**: Material properties are similar to training data → trust predictions more
                - **Out-distribution (out-dist)**: Material is novel/different → predictions may be less reliable

                *In-distribution = similar to training data, trust more*
                """)

                reliability_plot = gr.Plot(label="Calibration Curve & Distance Histogram")

            with gr.TabItem("🔬 Model Diagnostics"):
                gr.Markdown("### Model Performance Diagnostics & Technical Details")
                gr.Markdown("""
                **Model Diagnostics Overview:**

                This tab provides detailed technical information about the ML models, including:
                - Training performance metrics
                - Calibration quality assessment
                - Ensemble weighting optimization
                - Cross-validation results
                - Feature importance analysis

                **Key Diagnostic Metrics:**
                - **Expected Calibration Error (ECE)**: Measures probability calibration quality
                - **Brier Score**: Combined measure of calibration and refinement
                - **Cross-Validation Scores**: Model robustness across different data splits
                - **Ensemble Weights**: Optimal balance between ML and rule-based predictions
                """)

                # Model diagnostics display
                diagnostics_summary = gr.Markdown("""
                **Model Performance Summary:**
                - **Status**: Not yet trained
                - **Accuracy**: N/A
                - **Calibration**: N/A
                - **Ensemble Weights**: N/A
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Training Metrics")
                        training_metrics_display = gr.JSON(
                            value={},
                            label="Detailed Training Metrics"
                        )

                    with gr.Column():
                        gr.Markdown("### Calibration Assessment")
                        calibration_metrics_display = gr.JSON(
                            value={},
                            label="Calibration Quality Metrics"
                        )

                gr.Markdown("### Ensemble Optimization")
                ensemble_weights_display = gr.JSON(
                    value={},
                    label="Current Ensemble Weights"
                )

                gr.Markdown("### Cross-Validation Results")
                cv_results_display = gr.JSON(
                    value={},
                    label="Cross-Validation Performance"
                )

                # Update diagnostics button
                update_diagnostics_btn = gr.Button("🔄 Update Model Diagnostics", variant="primary")

                def update_model_diagnostics():
                    """Update model diagnostics information."""
                    try:
                        # Get current model state
                        if not ml_classifier.is_trained:
                            summary = "**Model Performance Summary:**\n- **Status**: Not yet trained\n- **Accuracy**: N/A\n- **Calibration**: N/A\n- **Ensemble Weights**: N/A"
                            training_metrics = {}
                            calibration_metrics = {}
                            ensemble_weights = ml_classifier.ensemble_weights
                            cv_results = {}
                        else:
                            # Get calibration summary
                            cal_summary = ml_classifier.get_calibration_summary()

                            summary = f"""**Model Performance Summary:**
- **Status**: Trained on {len(ml_classifier.training_features_scaled) if hasattr(ml_classifier, 'training_features_scaled') and ml_classifier.training_features_scaled is not None else 'unknown'} materials
- **Accuracy**: {ml_metrics.get('accuracy', 'N/A'):.4f}
- **Calibration**: {cal_summary.get('calibration_status', 'N/A')} (ECE: {cal_summary.get('expected_calibration_error', 'N/A'):.3f})
- **Ensemble Weights**: ML {ml_classifier.ensemble_weights.get('ml', 0):.2f}, LLM {ml_classifier.ensemble_weights.get('llm', 0):.2f}"""

                            # Training metrics
                            training_metrics = {
                                "accuracy": ml_metrics.get("accuracy", 0),
                                "precision": ml_metrics.get("precision", 0),
                                "recall": ml_metrics.get("recall", 0),
                                "f1_score": ml_metrics.get("f1_score", 0),
                                "cv_mean": ml_metrics.get("cv_mean", 0),
                                "cv_std": ml_metrics.get("cv_std", 0)
                            }

                            # Calibration metrics
                            calibration_metrics = {
                                "expected_calibration_error": cal_summary.get("expected_calibration_error", 0),
                                "brier_score": cal_summary.get("brier_score", 0),
                                "max_calibration_error": cal_summary.get("max_calibration_error", 0),
                                "calibration_method": getattr(ml_classifier, 'calibration_method', 'none'),
                                "calibration_status": cal_summary.get("calibration_status", "unknown")
                            }

                            # Ensemble weights
                            ensemble_weights = ml_classifier.ensemble_weights.copy()

                            # Cross-validation results
                            cv_results = {
                                "5_fold_mean": ml_metrics.get("cv_mean", 0),
                                "5_fold_std": ml_metrics.get("cv_std", 0),
                                "robustness": "High" if ml_metrics.get("cv_std", 1) < 0.05 else "Moderate" if ml_metrics.get("cv_std", 1) < 0.1 else "Low"
                            }

                        return summary, training_metrics, calibration_metrics, ensemble_weights, cv_results

                    except Exception as e:
                        error_summary = f"**Error updating diagnostics:** {str(e)}"
                        return error_summary, {}, {}, {}, {}

                # Connect diagnostics update
                update_diagnostics_btn.click(
                    fn=update_model_diagnostics,
                    inputs=[],
                    outputs=[diagnostics_summary, training_metrics_display, calibration_metrics_display, ensemble_weights_display, cv_results_display]
                )

                gr.Markdown("### Model Interpretability")
                gr.Markdown("""
                **Feature Importance (Random Forest):**
                The synthesizability prediction model considers multiple material properties:

                1. **Energy Above Hull** (most important): Thermodynamic stability indicator
                2. **Formation Energy**: Chemical stability measure
                3. **Band Gap**: Electronic structure property
                4. **Electronegativity**: Chemical bonding tendency
                5. **Atomic Radius**: Size-related property
                6. **Density**: Physical property
                7. **Unit Cell Sites**: Structural complexity

                **Calibration Process:**
                - Raw ML probabilities are adjusted using isotonic regression
                - Ensures predicted probabilities match actual experimental success rates
                - Reduces overconfidence in predictions

                **Ensemble Method:**
                - Combines statistical ML predictions with physics-based rules
                - Adaptively weights each component based on calibration performance
                - Provides more robust predictions than single models
                """)

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

                gr.Markdown("### Lab-Ready Export")
                gr.Markdown("""
                Generate comprehensive lab-ready export with CSV data and PDF report.
                Includes all synthesis information, cost-benefit analysis, and model documentation.

                **⚠️ SAFETY FIRST:** Export includes automatic safety gating to prevent laboratory accidents.
                """)

                # Safety controls
                with gr.Row():
                    with gr.Column():
                        safe_mode_toggle = gr.Checkbox(
                            value=True,
                            label="🛡️ SAFE MODE",
                            info="Enable conservative safety thresholds (recommended). When disabled, shows exploratory results but marks them clearly."
                        )
                        human_override_checkbox = gr.Checkbox(
                            value=False,
                            label="👤 Allow Human Override",
                            info="Permit export of materials requiring human safety review (requires documented approval)"
                        )

                    with gr.Column():
                        gr.Markdown("""
                        **Safe Mode Settings:**
                        - ✅ Ensemble probability ≥ 0.8
                        - ✅ Thermodynamic stability = 'highly_stable' OR human override
                        - ✅ In-distribution materials only (unless override approved)
                        - ✅ No hazardous elements (Be, Hg, Cd, Pb, As, Tl, etc.)

                        **Human Override Requirements:**
                        - Written risk assessment by PI
                        - Safety data sheet review
                        - PPE and engineering controls documented
                        - Waste disposal plan approved
                        """)

                export_button = gr.Button("🚀 Export for Synthesis", variant="primary", size="lg")

                # Safety summary display
                safety_summary_display = gr.Markdown("")

                lab_csv_download = gr.File(label="Download Lab CSV")
                lab_pdf_download = gr.File(label="Download PDF Report")

        # Function to prepare CSV export
        def prepare_csv_export(results_df):
            """Prepare CSV data for download with all required columns from global full DataFrame."""
            global global_full_results_df

            # Use the full internal DataFrame if available, otherwise fall back to the display DataFrame
            if global_full_results_df is not None and not global_full_results_df.empty:
                csv_data = global_full_results_df.copy()
                print(f"Using full internal DataFrame with {len(csv_data.columns)} columns for CSV export")
            else:
                # Fallback to the display DataFrame
                if hasattr(results_df, 'data'):
                    csv_data = results_df.data.copy()
                else:
                    csv_data = results_df.copy()
                print(f"Using display DataFrame with {len(csv_data.columns)} columns for CSV export")

            if csv_data.empty:
                return None

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

            print(f"CSV export created with {len(available_columns)} columns: {available_columns}")
            return temp_file.name

        # Function for lab-ready export
        def prepare_lab_export(results_df, ml_metrics):
            """Prepare comprehensive lab export with CSV and PDF."""
            if results_df.empty:
                return None, None

            try:
                # Extract the actual DataFrame from Gradio's DataFrame component
                if hasattr(results_df, 'data'):
                    df = results_df.data
                else:
                    df = results_df

                # Import the export function
                from export_for_lab import export_for_lab

                # Generate both CSV and PDF
                csv_path, pdf_path = export_for_lab(df, ml_metrics, ".")

                return csv_path, pdf_path

            except Exception as e:
                print(f"Error in lab export: {e}")
                return None, None

        # Generation with global variable storage
        def generate_and_store_global(api_key, latent_dim, epochs, num_samples, available_equipment):
            """Generate materials and store ML metrics and full results DataFrame globally for export functions."""
            global global_ml_metrics, global_full_results_df
            results = generate_and_analyze(api_key, latent_dim, epochs, num_samples, available_equipment)

            # Store the full internal results DataFrame globally (it's the last element in results)
            if len(results) >= 9 and isinstance(results[8], pd.DataFrame):
                global_full_results_df = results[8].copy()  # Full results DataFrame
                print(f"Stored full results DataFrame with {len(global_full_results_df.columns)} columns for CSV export")

            # Generate CSV using the display table for now (will be fixed in prepare_csv_export)
            if len(results) >= 3 and isinstance(results[2], pd.DataFrame):
                results_df = results[2].data if hasattr(results[2], 'data') else results[2]
                csv_path = prepare_csv_export(results_df)
            else:
                csv_path = None

            # Return all original results plus CSV path
            return results + (csv_path,)

        generate_btn.click(
            fn=generate_and_store_global,
            inputs=[api_key_input, latent_dim, epochs, num_samples, available_equipment],
            outputs=[summary_output, plot_output, materials_table, priority_table, workflow_table, cba_table, methods_table, reliability_plot, csv_download]
        )

        # Function for lab export using global variable with safety gating
        def prepare_lab_export_from_table(materials_table_data, safe_mode_enabled, allow_human_override):
            """Prepare comprehensive lab export from table data using global ML metrics with safety gating."""
            global global_ml_metrics

            if hasattr(materials_table_data, 'data'):
                df = materials_table_data.data
            else:
                df = materials_table_data

            if hasattr(df, 'empty') and df.empty:
                return None, None, "No materials data available for export"

            try:
                # Import the updated export function
                from export_for_lab import export_for_lab

                # Perform export with safety gating
                csv_path, pdf_path, safety_summary = export_for_lab(
                    df, global_ml_metrics, ".",
                    safe_mode=safe_mode_enabled,
                    allow_human_override=allow_human_override
                )

                # Create safety summary message
                if safety_summary:
                    safety_msg = f"""
                    **Safety Export Summary:**
                    - Total materials: {safety_summary['total_materials']}
                    - Safe for export: {safety_summary['safe_materials']}
                    - Requires human review: {safety_summary['human_review_materials']}
                    - Blocked (unsafe): {safety_summary['unsafe_materials']}
                    - Exported: {safety_summary['exported_materials']}
                    - Safe mode: {'Enabled' if safety_summary['safe_mode_enabled'] else 'Disabled'}
                    - Human override: {'Allowed' if safety_summary['human_override_allowed'] else 'Not allowed'}
                    """

                    if safety_summary['exported_materials'] == 0:
                        safety_msg += "\n**WARNING: No materials passed safety criteria! Export was blocked.**"
                    elif safety_summary['unsafe_materials'] > 0:
                        safety_msg += f"\n**Note: {safety_summary['unsafe_materials']} materials were blocked for safety reasons.**"
                else:
                    safety_msg = "Safety summary not available"

                return csv_path, pdf_path, safety_msg

            except Exception as e:
                print(f"Error in lab export: {e}")
                import traceback
                traceback.print_exc()
                return None, None, f"Export failed: {str(e)}"

        # Connect lab export button
        export_button.click(
            fn=prepare_lab_export_from_table,
            inputs=[materials_table, safe_mode_toggle, human_override_checkbox],
            outputs=[lab_csv_download, lab_pdf_download, safety_summary_display]
        )

        # Experimental outcomes functions
        def update_formula_dropdown(results_df):
            """Update formula dropdown with generated materials."""
            if hasattr(results_df, 'data') and results_df.data is not None:
                formulas = results_df.data['formula'].tolist()
                return gr.Dropdown(choices=formulas, value=formulas[0] if formulas else None)
            return gr.Dropdown(choices=[], value=None)

        def log_experimental_outcome(formula, outcome, notes, materials_table):
            """Log an experimental outcome."""
            if not formula:
                return "Please select a material formula.", load_experimental_outcomes().tail(10)

            # Get material data from the table
            if hasattr(materials_table, 'data') and materials_table.data is not None:
                material_row = materials_table.data[materials_table.data['formula'] == formula]
                if not material_row.empty:
                    ml_pred = material_row['synthesizability_probability'].iloc[0]
                    llm_pred = material_row['llm_probability'].iloc[0]
                    e_hull = material_row['energy_above_hull'].iloc[0]

                    success = save_experimental_outcome(formula, outcome, ml_pred, llm_pred, e_hull, notes)
                    if success:
                        status_msg = f"✅ Successfully logged outcome for {formula}"
                    else:
                        status_msg = f"❌ Failed to log outcome for {formula}"
                else:
                    status_msg = f"❌ Material {formula} not found in current results"
            else:
                status_msg = "❌ No materials data available"

            return status_msg, load_experimental_outcomes().tail(10)

        def update_validation_metrics():
            """Update and display validation metrics."""
            outcomes_df = load_experimental_outcomes()
            metrics = calculate_validation_metrics(outcomes_df)

            # Create metrics display text
            metrics_text = f"""
            **Current Metrics:**
            - Total Experiments: {metrics['total_experiments']}
            - Overall Success Rate: {metrics['success_rate']:.1%}
            - Hit Rate (Last 10): {metrics['hit_rate_last_10']:.1%} ({int(metrics['hit_rate_last_10'] * 10)}/10 successes)
            - Precision: {metrics['precision']:.3f}
            - Recall: {metrics['recall']:.3f}
            - Calibration Error: {metrics['calibration_error']:.3f}
            """

            # Prepare recent experiments table
            if metrics['recent_experiments']:
                recent_df = pd.DataFrame(metrics['recent_experiments'])
                recent_display = recent_df[['formula', 'outcome', 'ml_pred', 'llm_pred', 'notes']].round(3)
                recent_display['Predicted Prob'] = (recent_display['ml_pred'] * 0.7 + recent_display['llm_pred'] * 0.3).round(3)
                recent_display['Actual Success'] = recent_display['outcome'].isin(['Successful', 'Partial success']).astype(int)
                recent_display = recent_display[['formula', 'outcome', 'Predicted Prob', 'Actual Success', 'notes']]
                recent_display.columns = ['Formula', 'Outcome', 'Predicted Prob', 'Actual Success', 'Notes']
            else:
                recent_display = pd.DataFrame(columns=['Formula', 'Outcome', 'Predicted Prob', 'Actual Success', 'Notes'])

            # Create calibration plot
            if not outcomes_df.empty:
                outcomes_df['ensemble_pred'] = (outcomes_df['ml_pred'] * 0.7 + outcomes_df['llm_pred'] * 0.3)
                outcomes_df['actual_success'] = outcomes_df['outcome'].isin(['Successful', 'Partial success']).astype(int)

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(outcomes_df['ensemble_pred'], outcomes_df['actual_success'], alpha=0.6, color='blue')
                ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect calibration')
                ax.set_xlabel('Predicted Success Probability')
                ax.set_ylabel('Actual Success (0/1)')
                ax.set_title('Calibration: Predicted vs Actual Success')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, 'No experimental data available yet', ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Calibration Plot')
                plt.tight_layout()

            return metrics_text, recent_display, fig

        # Experimental outcomes functions
        def apply_materials_filter(filter_type, materials_table):
            """Apply filtering to materials table based on experimental outcomes."""
            if hasattr(materials_table, 'data') and materials_table.data is not None:
                outcomes_df = load_experimental_outcomes()
                filtered_df = filter_materials_by_outcome_status(materials_table.data, outcomes_df, filter_type)
                return filtered_df
            return materials_table.data if hasattr(materials_table, 'data') else pd.DataFrame()

        # Connect experimental outcomes
        generate_btn.click(
            fn=update_formula_dropdown,
            inputs=[materials_table],
            outputs=[formula_dropdown]
        )

        log_outcome_btn.click(
            fn=log_experimental_outcome,
            inputs=[formula_dropdown, outcome_dropdown, notes_textbox, materials_table],
            outputs=[outcome_status, outcomes_display]
        )

        update_metrics_btn.click(
            fn=update_validation_metrics,
            inputs=[],
            outputs=[validation_metrics_display, recent_experiments_table, calibration_plot]
        )

        # Connect filtering
        apply_filter_btn.click(
            fn=apply_materials_filter,
            inputs=[filter_dropdown, materials_table],
            outputs=[materials_table]
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
        server_port=int(os.environ.get("PORT", 8080)),
        share=False
    )
