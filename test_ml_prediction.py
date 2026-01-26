#!/usr/bin/env python3
"""
Test ML prediction with field-mapped data
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_ml_prediction_with_field_mapping():
    """Test ML prediction with properly field-mapped data"""
    print("=== Testing ML Prediction with Field Mapping ===")
    
    try:
        # Create test data with all required fields (simulating field mapping output)
        test_data = {
            'element_1': ['Al', 'Ti', 'Fe'],
            'element_2': ['Ni', 'Cu', 'Co'],
            'composition_1': [0.5, 0.6, 0.7],
            'composition_2': [0.5, 0.4, 0.3],
            'formula': ['Al0.5Ni0.5', 'Ti0.6Cu0.4', 'Fe0.7Co0.3'],
            'melting_point': [1000, 1200, 1400],
            'density': [2.7, 4.5, 7.8],
            'electronegativity': [1.5, 1.8, 1.9],
            'atomic_radius': [1.43, 1.45, 1.25],
            'formation_energy_per_atom': [-1.2, -0.8, -1.5],
            'energy_above_hull': [0.02, 0.05, 0.01],
            'band_gap': [0.0, 0.1, 0.0],
            'nsites': [2, 2, 2],
            'is_generated': [True, True, True]
        }
        
        df = pd.DataFrame(test_data)
        print(f"Test DataFrame columns: {list(df.columns)}")
        print(f"Test DataFrame shape: {df.shape}")
        print("Test DataFrame:")
        print(df)
        print()
        
        # Test ML classifier
        from synthesizability_predictor import SynthesizabilityClassifier
        
        # Create a classifier instance
        ml_classifier = SynthesizabilityClassifier()
        
        # Train with synthetic data (since we don't have API key)
        print("Training ML classifier with synthetic data...")
        
        # Create synthetic training data
        from gradio_app import create_synthetic_dataset
        synthetic_data = create_synthetic_dataset(100)
        
        # Add required fields to synthetic data
        from centralized_field_mapping import apply_field_mapping_to_generation
        synthetic_data = apply_field_mapping_to_generation(synthetic_data)
        
        print(f"Synthetic training data shape: {synthetic_data.shape}")
        print(f"Synthetic training data columns: {list(synthetic_data.columns)}")
        
        # Train the classifier
        ml_metrics = ml_classifier.train(api_key=None)  # Will use synthetic data
        print(f"ML classifier trained successfully!")
        print(f"Training metrics: {ml_metrics}")
        
        # Test prediction on our test data
        print("\nTesting prediction on test data...")
        results = ml_classifier.predict(df)
        
        print(f"ML prediction results shape: {results.shape}")
        print(f"ML prediction results columns: {list(results.columns)}")
        print("ML prediction results:")
        print(results)
        
        # Check if prediction succeeded
        if 'synthesizability_probability' in results.columns:
            print(f"\n✅ ML prediction successful!")
            print(f"Predicted probabilities: {results['synthesizability_probability'].tolist()}")
            return True
        else:
            print(f"\n❌ ML prediction failed - missing required columns")
            return False
        
    except Exception as e:
        print(f"Error in ML prediction test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_generation_flow():
    """Test the complete generation flow with field mapping"""
    print("\n=== Testing Full Generation Flow ===")
    
    try:
        # Import the generation function
        from gradio_app import generate_materials, train_vae_model, create_synthetic_dataset
        from sklearn.preprocessing import StandardScaler
        
        # Create synthetic dataset for VAE training
        print("Creating synthetic dataset for VAE training...")
        dataset = create_synthetic_dataset(100)
        
        # Prepare features for VAE
        feature_cols = ['composition_1', 'composition_2', 'formation_energy_per_atom', 'density', 'electronegativity', 'atomic_radius']
        features = dataset[feature_cols].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        print(f"VAE training features shape: {features_scaled.shape}")
        
        # Train VAE
        print("Training VAE model...")
        vae_model = train_vae_model(features_scaled, latent_dim=5, epochs=10)  # Shorter training for test
        print("VAE model trained successfully!")
        
        # Generate materials
        print("Generating materials...")
        generated_df = generate_materials(vae_model, scaler, num_samples=5)
        
        print(f"Generated materials shape: {generated_df.shape}")
        print(f"Generated materials columns: {list(generated_df.columns)}")
        print("Generated materials:")
        print(generated_df)
        
        # Test that field mapping was applied
        required_fields = ['formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'nsites', 'density', 'electronegativity', 'atomic_radius']
        missing_fields = [field for field in required_fields if field not in generated_df.columns]
        
        if missing_fields:
            print(f"❌ Field mapping failed - missing fields: {missing_fields}")
            return False
        else:
            print(f"✅ Field mapping successful - all required fields present")
            return True
        
    except Exception as e:
        print(f"Error in full generation flow test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting ML prediction tests...")
    
    success1 = test_ml_prediction_with_field_mapping()
    success2 = test_full_generation_flow()
    
    if success1 and success2:
        print("\n✅ All tests passed!")
        print("The field mapping and ML prediction system is working correctly!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)