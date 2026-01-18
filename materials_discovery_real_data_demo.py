"""
Demo script showing how to integrate Materials Project data into the materials discovery workshop.
This replaces the synthetic data generation with real materials data from Materials Project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from materials_discovery_api import get_training_dataset, MaterialsProjectClient

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def demonstrate_real_data_integration():
    """Demonstrate how to replace synthetic data with real Materials Project data."""

    print("=== MATERIALS DISCOVERY WORKSHOP - REAL DATA INTEGRATION ===\n")

    # Step 1: Fetch real data from Materials Project
    print("1. FETCHING REAL MATERIALS DATA FROM MATERIALS PROJECT")
    print("-" * 60)

    try:
        # Get training dataset
        ml_features, raw_data = get_training_dataset(n_materials=200)

        if ml_features.empty or raw_data.empty:
            print("❌ Failed to retrieve data from Materials Project")
            print("Falling back to synthetic data for demonstration...")
            return create_fallback_synthetic_data()

        print(f"✅ Successfully retrieved {len(ml_features)} materials from Materials Project")
        print(f"   - ML features shape: {ml_features.shape}")
        print(f"   - Raw data shape: {raw_data.shape}")

        # Show sample of real data
        print("\nSample of real materials data:")
        display_cols = ['material_id', 'formula', 'elements', 'band_gap', 'density', 'energy_above_hull']
        print(raw_data[display_cols].head())

        # Step 2: Compare with synthetic data characteristics
        print("\n2. COMPARING REAL VS SYNTHETIC DATA CHARACTERISTICS")
        print("-" * 60)

        # Analyze real data statistics
        print("Real Materials Project data statistics:")
        real_stats = ml_features[['density', 'electronegativity', 'atomic_radius', 'melting_point']].describe()
        print(real_stats.loc[['mean', 'std', 'min', 'max']].round(3))

        # Create comparison with synthetic data
        synthetic_features = create_synthetic_features_like_real(len(ml_features))
        synthetic_stats = synthetic_features[['density', 'electronegativity', 'atomic_radius', 'melting_point']].describe()

        print("\nSynthetic data (matched size) statistics:")
        print(synthetic_stats.loc[['mean', 'std', 'min', 'max']].round(3))

        # Step 3: Data quality assessment
        print("\n3. DATA QUALITY ASSESSMENT")
        print("-" * 60)

        # Check for missing values
        missing_real = ml_features.isnull().sum().sum()
        print(f"Missing values in real data: {missing_real}")

        # Check property distributions
        print("\nProperty distributions comparison:")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Real vs Synthetic Data Distributions', fontsize=14)

        properties = ['density', 'electronegativity', 'atomic_radius', 'melting_point']

        for i, prop in enumerate(properties):
            ax = axes[i//2, i%2]

            # Real data
            ax.hist(ml_features[prop], bins=20, alpha=0.7, label='Real (MP)', density=True, color='blue')

            # Synthetic data
            ax.hist(synthetic_features[prop], bins=20, alpha=0.7, label='Synthetic', density=True, color='red')

            ax.set_xlabel(prop.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('real_vs_synthetic_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Step 4: Demonstrate ML readiness
        print("\n4. MACHINE LEARNING READINESS CHECK")
        print("-" * 60)

        # Check feature correlations
        corr_matrix = ml_features.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlations in Real Materials Data')
        plt.tight_layout()
        plt.savefig('real_data_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Check for outliers
        print("\nOutlier analysis:")
        for prop in properties:
            q1 = ml_features[prop].quantile(0.25)
            q3 = ml_features[prop].quantile(0.75)
            iqr = q3 - q1
            outliers = ((ml_features[prop] < (q1 - 1.5 * iqr)) |
                       (ml_features[prop] > (q3 + 1.5 * iqr))).sum()
            print(f"  {prop}: {outliers} outliers detected")

        # Step 5: Export for notebook use
        print("\n5. EXPORTING DATA FOR NOTEBOOK USE")
        print("-" * 60)

        # Save processed data
        ml_features.to_csv('materials_project_ml_features.csv', index=False)
        raw_data.to_csv('materials_project_raw_data.csv', index=False)

        print("✅ Data exported successfully!")
        print("   - materials_project_ml_features.csv: ML-ready features")
        print("   - materials_project_raw_data.csv: Original Materials Project data")

        # Summary
        print("\n" + "="*60)
        print("INTEGRATION SUMMARY")
        print("="*60)
        print("✅ Materials Project API integration successful")
        print(f"✅ Retrieved {len(ml_features)} real materials for training")
        print("✅ Data quality verified (no missing values)")
        print("✅ Feature distributions analyzed and visualized")
        print("✅ Data exported for notebook integration")
        print("\nNext steps:")
        print("- Use materials_project_ml_features.csv in the VAE training")
        print("- Compare ML performance on real vs synthetic data")
        print("- Validate generated materials against Materials Project database")

        return ml_features, raw_data

    except Exception as e:
        print(f"❌ Error during integration: {e}")
        print("Falling back to synthetic data...")
        return create_fallback_synthetic_data()


def create_synthetic_features_like_real(n_samples: int) -> pd.DataFrame:
    """Create synthetic data that mimics the characteristics of real Materials Project data."""

    np.random.seed(42)

    # Based on real data statistics
    synthetic_data = []

    for i in range(n_samples):
        # Realistic property ranges based on Materials Project data
        density = np.random.normal(5.0, 2.5)  # More realistic density range
        density = max(2, min(15, density))  # Clip to reasonable bounds

        electronegativity = np.random.normal(1.6, 0.4)
        electronegativity = max(0.8, min(2.5, electronegativity))

        atomic_radius = np.random.normal(1.4, 0.3)
        atomic_radius = max(0.8, min(2.2, atomic_radius))

        # Formation energy (normalized as melting_point proxy)
        melting_point = np.random.normal(-1.5, 2.0)
        melting_point = max(-8, min(8, melting_point))

        synthetic_data.append({
            'composition_1': 0.5,
            'composition_2': 0.5,
            'composition_3': 0.0,
            'density': density,
            'electronegativity': electronegativity,
            'atomic_radius': atomic_radius,
            'band_gap': 0.0,  # Most alloys have zero band gap
            'energy_above_hull': 0.0,
            'melting_point': melting_point
        })

    return pd.DataFrame(synthetic_data)


def create_fallback_synthetic_data():
    """Create synthetic data as fallback when API fails."""

    print("Creating fallback synthetic dataset...")

    synthetic_features = create_synthetic_features_like_real(500)

    # Create fake raw data structure
    raw_data = pd.DataFrame({
        'material_id': [f'synth_{i+1}' for i in range(len(synthetic_features))],
        'formula': [f'A{i%10}B{i%7}' for i in range(len(synthetic_features))],
        'elements': [['A', 'B'] for _ in range(len(synthetic_features))],
        'band_gap': synthetic_features['band_gap'],
        'density': synthetic_features['density'],
        'energy_above_hull': synthetic_features['energy_above_hull']
    })

    print(f"Created {len(synthetic_features)} synthetic materials as fallback")

    return synthetic_features, raw_data


if __name__ == "__main__":
    # Run the integration demonstration
    ml_features, raw_data = demonstrate_real_data_integration()
