#!/usr/bin/env python3
"""
Test script for synthesizability prediction functionality
This demonstrates the complete end-to-end materials discovery pipeline
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Mock ICSD data for demonstration (in practice, this would come from actual ICSD database)
def generate_mock_icsd_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate mock ICSD data representing experimentally synthesized materials."""
    np.random.seed(42)

    data = []
    for i in range(n_samples):
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

        data.append({
            'formation_energy_per_atom': formation_energy,
            'band_gap': band_gap,
            'energy_above_hull': energy_above_hull,
            'electronegativity': electronegativity,
            'atomic_radius': atomic_radius,
            'nsites': nsites,
            'density': density,
            'synthesizable': 1
        })

    return pd.DataFrame(data)

def generate_mock_mp_only_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate mock MP-only data representing theoretically predicted materials."""
    np.random.seed(123)

    data = []
    for i in range(n_samples):
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

        data.append({
            'formation_energy_per_atom': formation_energy,
            'band_gap': band_gap,
            'energy_above_hull': energy_above_hull,
            'electronegativity': electronegativity,
            'atomic_radius': atomic_radius,
            'nsites': nsites,
            'density': density,
            'synthesizable': 0
        })

    return pd.DataFrame(data)

class SynthesizabilityClassifier:
    """ML-based classifier for predicting material synthesizability."""

    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'formation_energy_per_atom', 'band_gap', 'energy_above_hull',
            'electronegativity', 'atomic_radius', 'nsites', 'density'
        ]
        self.is_trained = False

    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from ICSD and MP-only materials."""
        icsd_data = generate_mock_icsd_data(1500)
        mp_only_data = generate_mock_mp_only_data(1500)

        training_data = pd.concat([icsd_data, mp_only_data], ignore_index=True)
        training_data = training_data.sample(frac=1, random_state=42).reset_index(drop=True)

        X = training_data[self.feature_columns]
        y = training_data['synthesizable']

        return X, y

    def train(self, test_size: float = 0.2):
        """Train the synthesizability classifier."""
        X, y = self.prepare_training_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

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

    def predict_single(self, material_properties: Dict) -> Dict:
        """Predict synthesizability for a single material."""
        df = pd.DataFrame([material_properties])
        result_df = self.predict(df)

        return {
            'prediction': int(result_df['synthesizability_prediction'].iloc[0]),
            'probability': float(result_df['synthesizability_probability'].iloc[0]),
            'confidence': float(result_df['synthesizability_confidence'].iloc[0])
        }

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

def thermodynamic_stability_check(materials_df: pd.DataFrame) -> pd.DataFrame:
    """Check thermodynamic stability of materials."""
    results_df = materials_df.copy()

    results_df['thermodynamically_stable'] = results_df['energy_above_hull'] <= 0.1
    results_df['formation_energy_reasonable'] = (
        (results_df['formation_energy_per_atom'] >= -10) &
        (results_df['formation_energy_per_atom'] <= 2)
    )
    results_df['overall_stability'] = (
        results_df['thermodynamically_stable'] &
        results_df['formation_energy_reasonable']
    )

    return results_df

def calculate_synthesis_priority(materials_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate priority ranking for experimental synthesis."""
    results_df = materials_df.copy()

    stability_score = 1 - results_df['energy_above_hull'].clip(0, 1)
    synth_score = results_df.get('synthesizability_probability', 0.5)
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
        'synthesis_priority_score', 'synthesizability_probability',
        'energy_above_hull', 'formation_energy_per_atom',
        'estimated_time_days', 'required_equipment'
    ]]

def initialize_synthesizability_models():
    """Initialize and train synthesizability prediction models."""
    print("Training synthesizability prediction models...")

    ml_classifier = SynthesizabilityClassifier(model_type='random_forest')
    metrics = ml_classifier.train()

    print("ML Classifier trained successfully!")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  CV Mean: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")

    llm_predictor = LLMSynthesizabilityPredictor()

    return ml_classifier, llm_predictor

def main():
    """Run the complete synthesizability prediction demonstration."""
    print("🧪 SYNTHESIZABILITY PREDICTION DEMONSTRATION")
    print("=" * 60)

    # Initialize models
    ml_classifier, llm_predictor = initialize_synthesizability_models()

    # Generate sample materials for testing
    print("\n🎨 Generating sample materials for testing...")

    test_materials = pd.DataFrame([
        {
            'formula': 'Al0.5Ti0.5',
            'formation_energy_per_atom': -1.5,
            'band_gap': 0.0,
            'energy_above_hull': 0.05,
            'electronegativity': 1.7,
            'atomic_radius': 1.4,
            'nsites': 4,
            'density': 4.5
        },
        {
            'formula': 'Fe0.5Co0.5',
            'formation_energy_per_atom': -0.2,
            'band_gap': 0.0,
            'energy_above_hull': 0.8,
            'electronegativity': 1.9,
            'atomic_radius': 1.3,
            'nsites': 32,
            'density': 8.2
        },
        {
            'formula': 'Zn0.5Cu0.5',
            'formation_energy_per_atom': 0.3,
            'band_gap': 0.1,
            'energy_above_hull': 0.02,
            'electronegativity': 1.8,
            'atomic_radius': 1.4,
            'nsites': 8,
            'density': 7.1
        }
    ])

    print(f"Generated {len(test_materials)} test materials")
    print(test_materials[['formula', 'energy_above_hull', 'density']].to_string(index=False))

    # Test thermodynamic stability
    print("\n=== THERMODYNAMIC STABILITY VALIDATION ===")
    stability_results = thermodynamic_stability_check(test_materials)
    print(f"Stable materials: {stability_results['overall_stability'].sum()}/{len(stability_results)}")
    print(f"  - Low E_hull: {stability_results['thermodynamically_stable'].sum()}")
    print(f"  - Reasonable formation energy: {stability_results['formation_energy_reasonable'].sum()}")

    # Filter to stable materials
    stable_materials = stability_results[stability_results['overall_stability']].copy()
    print(f"\nProceeding with {len(stable_materials)} stable materials")

    if len(stable_materials) == 0:
        print("No stable materials found - using all materials for demonstration")
        stable_materials = stability_results.copy()

    # ML predictions
    print("\n=== SYNTHESIZABILITY PREDICTIONS ===")
    ml_predictions = ml_classifier.predict(stable_materials)

    # LLM predictions
    llm_predictions = []
    for idx, material in stable_materials.iterrows():
        llm_result = llm_predictor.predict_synthesizability(material.to_dict())
        llm_predictions.append(llm_result)

    # Combine predictions
    results = ml_predictions.copy()
    results['llm_prediction'] = [p['prediction'] for p in llm_predictions]
    results['llm_probability'] = [p['probability'] for p in llm_predictions]
    results['llm_confidence'] = [p['confidence'] for p in llm_predictions]

    # Ensemble prediction
    results['ensemble_probability'] = (
        0.7 * results['synthesizability_probability'] +
        0.3 * results['llm_probability']
    )
    results['ensemble_prediction'] = (results['ensemble_probability'] >= 0.5).astype(int)
    results['ensemble_confidence'] = np.abs(results['ensemble_probability'] - 0.5) * 2

    print("\nPrediction Results:")
    display_cols = ['formula', 'synthesizability_probability', 'llm_probability',
                   'ensemble_probability', 'ensemble_prediction']
    print(results[display_cols].to_string(index=False))

    # Priority ranking
    print("\n=== SYNTHESIS PRIORITY RANKING ===")
    priority_results = calculate_synthesis_priority(results)
    top_candidates = priority_results.head(3)[['formula', 'synthesis_priority_score',
                                             'ensemble_probability', 'energy_above_hull']]
    print("Top synthesis candidates:")
    print(top_candidates.to_string(index=False))

    # Experimental workflow
    print("\n=== EXPERIMENTAL WORKFLOW ===")
    workflow = create_experimental_workflow(results, top_n=3)
    print("Prioritized workflow:")
    print(workflow[['workflow_step', 'priority_level', 'synthesis_priority_score',
                   'estimated_time_days']].to_string(index=False))

    print("\n" + "=" * 60)
    print("✅ SYNTHESIZABILITY PREDICTION DEMONSTRATION COMPLETED!")
    print("🧪 Materials discovery pipeline: Generation → Stability → Synthesizability → Experiment")
    print("=" * 60)

if __name__ == "__main__":
    main()
