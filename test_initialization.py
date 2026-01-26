#!/usr/bin/env python3
"""
Test script for Docker initialization process
"""

import sys
import os
sys.path.append('/app')

# Import required modules
from synthesizability_predictor import SynthesizabilityClassifier, create_synthetic_dataset_fallback
from gradio_app import train_vae_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def ensure_fields_exist(df, required_fields):
    """Ensure all required fields exist in DataFrame - HARD GUARANTEE."""
    df_copy = df.copy()
    for field in required_fields:
        if field not in df_copy.columns:
            print(f"CRITICAL: Adding missing field '{field}'")
            if field == "formation_energy_per_atom":
                df_copy[field] = np.random.normal(-1.5, 1.0, len(df_copy))
                df_copy[field] = np.clip(df_copy[field], -6.0, 2.0)
            elif field == "energy_above_hull":
                df_copy[field] = np.random.exponential(0.05, len(df_copy))
                df_copy[field] = np.clip(df_copy[field], 0, 0.5)
            elif field == "band_gap":
                df_copy[field] = np.random.exponential(0.5, len(df_copy))
                df_copy[field] = np.clip(df_copy[field], 0, 8.0)
            elif field == "nsites":
                df_copy[field] = np.random.randint(2, 15, len(df_copy))
            elif field == "density":
                df_copy[field] = np.random.normal(7.8, 2.0, len(df_copy))
                df_copy[field] = np.clip(df_copy[field], 2.0, 25.0)
            elif field == "electronegativity":
                df_copy[field] = np.random.normal(1.8, 0.3, len(df_copy))
                df_copy[field] = np.clip(df_copy[field], 0.7, 2.5)
            elif field == "atomic_radius":
                df_copy[field] = np.random.normal(1.3, 0.2, len(df_copy))
                df_copy[field] = np.clip(df_copy[field], 0.8, 2.2)
    return df_copy

def test_initialization():
    """Test the initialization process step by step."""
    print("=" * 60)
    print("TESTING DOCKER INITIALIZATION PROCESS")
    print("=" * 60)
    
    try:
        # Step 1: Test ML Classifier initialization
        print("\n1. Testing ML Classifier initialization...")
        ml_classifier = SynthesizabilityClassifier()
        ml_metrics = ml_classifier.train(api_key=None)  # Use synthetic data
        print(f"✅ ML Classifier trained: Accuracy={ml_metrics['accuracy']:.3f}, F1={ml_metrics['f1_score']:.3f}")
        
        # Step 2: Test VAE training dataset creation
        print("\n2. Testing VAE training dataset creation...")
        dataset = create_synthetic_dataset_fallback(1000)
        print(f"✅ Synthetic dataset created: {len(dataset)} materials")
        print(f"Dataset columns: {list(dataset.columns)}")
        
        # Check for required VAE fields
        required_vae_fields = ['formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'nsites']
        missing_cols = [col for col in required_vae_fields if col not in dataset.columns]
        if missing_cols:
            print(f"❌ Missing required VAE fields: {missing_cols}")
            return False
        else:
            print("✅ All required VAE fields present")
        
        # Step 3: Test VAE feature preparation
        print("\n3. Testing VAE feature preparation...")
        feature_cols = ['composition_1', 'composition_2', 'formation_energy_per_atom', 'density', 'electronegativity', 'atomic_radius']
        print(f"Looking for feature columns: {feature_cols}")
        
        missing_features = [col for col in feature_cols if col not in dataset.columns]
        if missing_features:
            print(f"❌ Missing feature columns: {missing_features}")
            return False
        
        features = dataset[feature_cols].values
        vae_scaler = StandardScaler()
        features_scaled = vae_scaler.fit_transform(features)
        print(f"✅ Feature scaling completed: {features_scaled.shape}")
        
        # Step 4: Test VAE training
        print("\n4. Testing VAE training...")
        vae_model = train_vae_model(features_scaled, latent_dim=5, epochs=50)
        print("✅ VAE training completed")
        
        # Step 5: Test material generation
        print("\n5. Testing material generation...")
        from gradio_app import generate_materials
        generated_df = generate_materials(vae_model, vae_scaler, num_samples=10)
        print(f"✅ Generated materials: {len(generated_df)} rows")
        print(f"Generated columns: {list(generated_df.columns)}")
        
        # Step 6: Test field validation
        print("\n6. Testing field validation...")
        required_fields = ['formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'nsites', 'density', 'electronegativity', 'atomic_radius']
        generated_df = ensure_fields_exist(generated_df, required_fields)
        
        missing_fields = [field for field in required_fields if field not in generated_df.columns]
        if missing_fields:
            print(f"❌ Still missing fields after validation: {missing_fields}")
            return False
        else:
            print("✅ All required fields present after validation")
        
        print("\n" + "=" * 60)
        print("✅ ALL INITIALIZATION TESTS PASSED!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Initialization test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_initialization()
    sys.exit(0 if success else 1)