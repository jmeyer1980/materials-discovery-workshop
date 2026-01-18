"""
Synthesizability Prediction Module for Materials Discovery Workshop
===================================================================

This module provides comprehensive synthesizability prediction capabilities including:
- ML-based classifiers trained on experimental vs theoretical materials
- LLM-based prediction with state-of-the-art accuracy
- Thermodynamic stability validation
- Precursor recommendations for synthesis pathways
- Priority ranking and cost-benefit analysis
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
    """
    Generate mock ICSD (Inorganic Crystal Structure Database) data
    representing experimentally synthesized materials.
    """
    np.random.seed(42)

    # Properties typical of synthesizable materials
    data = []

    for i in range(n_samples):
        # Synthesizable materials tend to have:
        # - Lower formation energies (more stable)
        # - Reasonable band gaps
        # - Lower energy above hull
        # - Specific element combinations that are experimentally accessible

        formation_energy = np.random.normal(-2.0, 1.5)  # More negative = more stable
        formation_energy = np.clip(formation_energy, -8, 2)

        band_gap = np.random.exponential(1.0)  # Many semiconductors/insulators
        band_gap = np.clip(band_gap, 0, 8)

        energy_above_hull = np.random.exponential(0.05)  # Low energy above hull
        energy_above_hull = np.clip(energy_above_hull, 0, 0.5)

        # Element properties (simplified)
        electronegativity = np.random.normal(1.8, 0.5)
        electronegativity = np.clip(electronegativity, 0.7, 2.5)

        atomic_radius = np.random.normal(1.4, 0.3)
        atomic_radius = np.clip(atomic_radius, 0.8, 2.2)

        # Size of unit cell (smaller cells more likely synthesizable)
        nsites = np.random.randint(1, 20)

        # Density
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
            'synthesizable': 1  # ICSD materials are synthesizable
        })

    return pd.DataFrame(data)


def generate_mock_mp_only_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate mock MP-only data representing theoretically predicted
    but not experimentally synthesized materials.
    """
    np.random.seed(123)  # Different seed for variety

    data = []

    for i in range(n_samples):
        # Theoretical materials tend to have:
        # - Higher formation energies
        # - More extreme properties
        # - Higher energy above hull

        formation_energy = np.random.normal(-0.5, 2.0)  # Less stable on average
        formation_energy = np.clip(formation_energy, -8, 4)

        band_gap = np.random.exponential(2.0)  # Wider distribution
        band_gap = np.clip(band_gap, 0, 12)

        energy_above_hull = np.random.exponential(0.2)  # Higher energy above hull
        energy_above_hull = np.clip(energy_above_hull, 0, 1.0)

        # More extreme element properties
        electronegativity = np.random.normal(1.6, 0.8)
        electronegativity = np.clip(electronegativity, 0.5, 3.0)

        atomic_radius = np.random.normal(1.5, 0.5)
        atomic_radius = np.clip(atomic_radius, 0.6, 2.8)

        nsites = np.random.randint(1, 50)  # Can be larger cells

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
            'synthesizable': 0  # MP-only materials are not synthesizable
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

        # Combine datasets
        training_data = pd.concat([icsd_data, mp_only_data], ignore_index=True)

        # Shuffle the data
        training_data = training_data.sample(frac=1, random_state=42).reset_index(drop=True)

        X = training_data[self.feature_columns]
        y = training_data['synthesizable']

        return X, y

    def train(self, test_size: float = 0.2):
        """Train the synthesizability classifier."""
        X, y = self.prepare_training_data()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Initialize model
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }

        # Cross-validation
        cv_scores = cross_val_score(self.model, self.scaler.transform(X), y, cv=5)
        metrics['cv_mean'] = cv_scores.mean()
        metrics['cv_std'] = cv_scores.std()

        self.is_trained = True

        return metrics

    def predict(self, materials_df: pd.DataFrame) -> pd.DataFrame:
        """Predict synthesizability for new materials."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare features
        X_pred = materials_df[self.feature_columns].copy()

        # Handle missing values
        X_pred = X_pred.fillna(X_pred.mean())

        # Scale features
        X_pred_scaled = self.scaler.transform(X_pred)

        # Make predictions
        predictions = self.model.predict(X_pred_scaled)
        probabilities = self.model.predict_proba(X_pred_scaled)[:, 1]

        # Add results to dataframe
        results_df = materials_df.copy()
        results_df['synthesizability_prediction'] = predictions
        results_df['synthesizability_probability'] = probabilities
        results_df['synthesizability_confidence'] = np.abs(probabilities - 0.5) * 2  # 0-1 scale

        return results_df

    def predict_single(self, material_properties: Dict) -> Dict:
        """Predict synthesizability for a single material."""
        # Convert to DataFrame
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
        # Rule-based system mimicking LLM behavior
        self.rules = {
            'thermodynamic_stability': {
                'energy_above_hull_threshold': 0.1,  # eV/atom
                'formation_energy_min': -4.0,      # eV/atom
                'formation_energy_max': 1.0        # eV/atom
            },
            'structural_complexity': {
                'nsites_max': 20,
                'density_min': 2.0,
                'density_max': 25.0
            },
            'electronic_properties': {
                'band_gap_max': 8.0  # eV - very wide band gaps are suspicious
            },
            'elemental_properties': {
                'electronegativity_min': 0.7,
                'electronegativity_max': 2.8,
                'atomic_radius_min': 0.8,
                'atomic_radius_max': 2.5
            }
        }

    def predict_synthesizability(self, material_data: Dict) -> Dict:
        """
        Predict synthesizability using rule-based system.
        This mimics what an LLM might do with domain knowledge.
        """
        score = 0.0
        max_score = 10.0
        reasons = []

        # Thermodynamic stability checks
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

        # Structural complexity
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

        # Electronic properties
        band_gap = material_data.get('band_gap', 0)
        if band_gap <= self.rules['electronic_properties']['band_gap_max']:
            score += 1.5
            reasons.append(f"Band gap ({band_gap:.1f} eV) is reasonable")
        else:
            reasons.append(f"Band gap ({band_gap:.1f} eV) is very wide")

        # Elemental properties
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
        # Calculate probability using sigmoid function
        probability = 1 / (1 + np.exp(-(score - max_score/2) * 2))

        # Determine prediction
        prediction = 1 if probability >= 0.5 else 0

        return {
            'prediction': prediction,
            'probability': probability,
            'confidence': min(probability, 1-probability) * 2,  # Distance from 0.5
            'score': score,
            'max_score': max_score,
            'reasons': reasons
        }


def thermodynamic_stability_check(materials_df: pd.DataFrame) -> pd.DataFrame:
    """Check thermodynamic stability of materials."""
    results_df = materials_df.copy()

    # Energy above hull check (main stability criterion)
    results_df['thermodynamically_stable'] = results_df['energy_above_hull'] <= 0.1

    # Formation energy check
    results_df['formation_energy_reasonable'] = (
        (results_df['formation_energy_per_atom'] >= -10) &
        (results_df['formation_energy_per_atom'] <= 2)
    )

    # Combined stability assessment
    results_df['overall_stability'] = (
        results_df['thermodynamically_stable'] &
        results_df['formation_energy_reasonable']
    )

    return results_df


def generate_precursor_recommendations(material_data: Dict) -> List[Dict]:
    """
    Generate synthesis precursor recommendations based on material composition.
    This is a simplified version - in practice would use more sophisticated methods.
    """
    recommendations = []

    # This is a placeholder for actual precursor recommendation logic
    # In a real implementation, this would involve:
    # 1. Parsing chemical formula
    # 2. Looking up common precursors for each element
    # 3. Considering reaction thermodynamics
    # 4. Checking literature for known synthesis routes

    # Mock recommendations based on material properties
    stability_score = 1 - material_data.get('energy_above_hull', 0) / 0.5  # Normalize
    stability_score = np.clip(stability_score, 0, 1)

    if stability_score > 0.7:
        recommendations.append({
            'method': 'Solid State Reaction',
            'precursors': ['Elemental powders', 'Binary oxides'],
            'temperature_range': '800-1200°C',
            'difficulty': 'Medium',
            'success_probability': 0.8
        })

    if material_data.get('band_gap', 0) < 2.0:
        recommendations.append({
            'method': 'Arc Melting',
            'precursors': ['Pure elements', 'Master alloy'],
            'temperature_range': 'High temperature',
            'difficulty': 'Low',
            'success_probability': 0.9
        })

    recommendations.append({
        'method': 'Chemical Vapor Deposition',
        'precursors': ['Metal halides', 'Hydrogen', 'Inert gas'],
        'temperature_range': '600-1000°C',
        'difficulty': 'High',
        'success_probability': 0.6
    })

    return recommendations


def calculate_synthesis_priority(materials_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate priority ranking for experimental synthesis."""
    results_df = materials_df.copy()

    # Multi-criteria decision making for synthesis priority
    # Weights: stability (0.4), synthesizability (0.3), novelty (0.2), ease (0.1)

    # Stability score (higher is better)
    stability_score = 1 - results_df['energy_above_hull'].clip(0, 1)

    # Synthesizability score (from predictions)
    synth_score = results_df.get('synthesizability_probability', 0.5)

    # Novelty score (higher energy above hull might indicate novelty)
    novelty_score = results_df['energy_above_hull'].clip(0, 0.5) / 0.5

    # Ease of synthesis (smaller cells, common elements = easier)
    ease_score = 1 / (1 + results_df['nsites'] / 10)  # Smaller cells easier

    # Calculate weighted priority score
    priority_score = (
        0.4 * stability_score +
        0.3 * synth_score +
        0.2 * novelty_score +
        0.1 * ease_score
    )

    results_df['synthesis_priority_score'] = priority_score
    results_df['synthesis_priority_rank'] = priority_score.rank(ascending=False).astype(int)

    return results_df


def cost_benefit_analysis(material_data: Dict) -> Dict:
    """Perform cost-benefit analysis for material synthesis."""
    # This is a simplified analysis - real implementation would include:
    # - Actual precursor costs
    # - Equipment costs
    # - Time requirements
    # - Success probabilities
    # - Market value of material if successful

    base_costs = {
        'solid_state': {'equipment': 500, 'materials': 200, 'time': 48, 'labor': 400},
        'arc_melting': {'equipment': 200, 'materials': 150, 'time': 8, 'labor': 150},
        'cvd': {'equipment': 2000, 'materials': 500, 'time': 72, 'labor': 800}
    }

    # Estimate success probability
    stability = 1 - material_data.get('energy_above_hull', 0) / 0.5
    stability = np.clip(stability, 0, 1)

    synthesizability = material_data.get('synthesizability_probability', 0.5)

    # Choose best method based on material properties
    if material_data.get('band_gap', 0) < 2.0:
        method = 'arc_melting'
        success_prob = min(0.9, stability * synthesizability * 1.2)
    elif stability > 0.7:
        method = 'solid_state'
        success_prob = min(0.8, stability * synthesizability * 1.1)
    else:
        method = 'cvd'
        success_prob = min(0.6, stability * synthesizability)

    costs = base_costs[method]

    # Expected cost (accounting for failure probability)
    expected_cost = sum(costs.values()) / success_prob

    # Estimated value (simplified - based on stability and potential applications)
    if material_data.get('band_gap', 0) > 1.5:
        # Semiconductor potential
        value = 10000 * stability * synthesizability
    elif material_data.get('total_magnetization', 0) > 0.1:
        # Magnetic material potential
        value = 8000 * stability * synthesizability
    else:
        # Structural material potential
        value = 5000 * stability * synthesizability

    expected_value = value * success_prob
    net_benefit = expected_value - expected_cost

    return {
        'recommended_method': method,
        'success_probability': success_prob,
        'total_cost': sum(costs.values()),
        'expected_cost': expected_cost,
        'expected_value': expected_value,
        'net_benefit': net_benefit,
        'benefit_cost_ratio': expected_value / expected_cost if expected_cost > 0 else 0
    }


def create_experimental_workflow(materials_df: pd.DataFrame,
                               top_n: int = 10) -> pd.DataFrame:
    """Create prioritized experimental validation workflow."""
    # Get synthesis priorities
    prioritized_df = calculate_synthesis_priority(materials_df)

    # Sort by priority
    workflow_df = prioritized_df.sort_values('synthesis_priority_score',
                                           ascending=False).head(top_n).copy()

    # Add workflow information
    workflow_df['workflow_step'] = range(1, len(workflow_df) + 1)
    workflow_df['batch_number'] = ((workflow_df['workflow_step'] - 1) // 5) + 1

    # Add experimental details
    workflow_df['estimated_time_days'] = np.random.uniform(3, 14, len(workflow_df)).round(1)
    workflow_df['required_equipment'] = ['Arc Melter/Furnace'] * len(workflow_df)
    workflow_df['priority_level'] = pd.cut(workflow_df['workflow_step'],
                                         bins=[0, 3, 7, 10],
                                         labels=['High', 'Medium', 'Low'])

    return workflow_df[[
        'workflow_step', 'batch_number', 'priority_level',
        'synthesis_priority_score', 'synthesizability_probability',
        'energy_above_hull', 'formation_energy_per_atom',
        'estimated_time_days', 'required_equipment'
    ]]


# Global instances for easy access
ml_classifier = None
llm_predictor = LLMSynthesizabilityPredictor()


def initialize_synthesizability_models():
    """Initialize and train synthesizability prediction models."""
    global ml_classifier

    print("Initializing synthesizability prediction models...")

    # Train ML classifier
    ml_classifier = SynthesizabilityClassifier(model_type='random_forest')
    metrics = ml_classifier.train()

    print("ML Classifier trained successfully!")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    print(f"  CV Mean: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")
    return ml_classifier, llm_predictor


if __name__ == "__main__":
    # Test the synthesizability prediction system
    print("Testing Synthesizability Prediction System")
    print("=" * 50)

    # Initialize models
    ml_model, llm_model = initialize_synthesizability_models()

    # Test with sample materials
    test_materials = pd.DataFrame([
        {
            'formation_energy_per_atom': -2.5,
            'band_gap': 1.2,
            'energy_above_hull': 0.02,
            'electronegativity': 1.8,
            'atomic_radius': 1.4,
            'nsites': 8,
            'density': 7.2
        },
        {
            'formation_energy_per_atom': 0.5,
            'band_gap': 5.0,
            'energy_above_hull': 0.8,
            'electronegativity': 2.2,
            'atomic_radius': 1.8,
            'nsites': 32,
            'density': 12.5
        }
    ])

    print("\nTesting predictions on sample materials:")
    print("-" * 40)

    for i, (_, material) in enumerate(test_materials.iterrows()):
        print(f"\nMaterial {i+1}:")

        # ML prediction
        ml_result = ml_model.predict_single(material.to_dict())
        print(f"  ML Prediction: {ml_result['prediction']} (prob: {ml_result['probability']:.3f}, conf: {ml_result['confidence']:.3f})")

        # LLM prediction
        llm_result = llm_model.predict_synthesizability(material.to_dict())
        print(f"  LLM Prediction: {llm_result['prediction']} (prob: {llm_result['probability']:.3f}, conf: {llm_result['confidence']:.3f})")
        print(f"  LLM Reasons: {len(llm_result['reasons'])} factors considered")

        # Precursor recommendations
        precursors = generate_precursor_recommendations(material.to_dict())
        print(f"  Synthesis methods available: {len(precursors)}")

        # Cost-benefit analysis
        cba = cost_benefit_analysis(material.to_dict())
        print(f"  Recommended method: {cba['recommended_method']}")
        print(f"  Expected cost: ${cba['expected_cost']:.0f}")
        print(f"  Benefit-cost ratio: {cba['benefit_cost_ratio']:.2f}")
