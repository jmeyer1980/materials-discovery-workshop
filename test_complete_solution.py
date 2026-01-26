#!/usr/bin/env python3
"""
Complete Solution Test
Tests the entire materials discovery system including:
- Field mapping
- ML prediction
- API authentication
- Synthetic dataset generation
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_api_authentication():
    """Test API authentication handler."""
    print("=== Testing API Authentication ===")
    
    try:
        from api_authentication_handler import APIAuthenticationHandler, validate_and_use_api_key
        
        # Test with no API key
        handler = APIAuthenticationHandler(None)
        status = handler.get_api_status()
        
        print(f"API key provided: {status['api_key_provided']}")
        print(f"API key valid: {status['api_key_valid']}")
        print(f"API status: {status['api_key_status']}")
        
        # Test validation function
        use_real_api = validate_and_use_api_key(None)
        print(f"Should use real API: {use_real_api}")
        
        return True
        
    except Exception as e:
        print(f"API authentication test failed: {e}")
        return False

def test_field_mapping():
    """Test field mapping functionality."""
    print("\n=== Testing Field Mapping ===")
    
    try:
        from centralized_field_mapping import apply_field_mapping_to_generation, REQUIRED_FIELDS
        
        # Create test data without required fields
        test_data = {
            'element_1': ['Al', 'Ti'],
            'element_2': ['Ni', 'Cu'],
            'composition_1': [0.5, 0.6],
            'composition_2': [0.5, 0.4],
            'formula': ['Al0.5Ni0.5', 'Ti0.6Cu0.4'],
            'melting_point': [1000, 1200],
            'density': [2.7, 4.5],
            'electronegativity': [1.5, 1.8],
            'atomic_radius': [1.43, 1.45]
        }
        
        df = pd.DataFrame(test_data)
        print(f"Before mapping: {list(df.columns)}")
        
        # Apply field mapping
        mapped_df = apply_field_mapping_to_generation(df)
        print(f"After mapping: {list(mapped_df.columns)}")
        
        # Check required fields
        missing_fields = [field for field in REQUIRED_FIELDS if field not in mapped_df.columns]
        if missing_fields:
            print(f"Missing fields: {missing_fields}")
            return False
        else:
            print("All required fields present after mapping")
            return True
            
    except Exception as e:
        print(f"Field mapping test failed: {e}")
        return False

def test_ml_prediction():
    """Test ML prediction with field-mapped data."""
    print("\n=== Testing ML Prediction ===")
    
    try:
        from synthesizability_predictor import SynthesizabilityClassifier
        
        # Create test data with all required fields
        test_data = {
            'element_1': ['Al', 'Ti'],
            'element_2': ['Ni', 'Cu'],
            'composition_1': [0.5, 0.6],
            'composition_2': [0.5, 0.4],
            'formula': ['Al0.5Ni0.5', 'Ti0.6Cu0.4'],
            'melting_point': [1000, 1200],
            'density': [2.7, 4.5],
            'electronegativity': [1.5, 1.8],
            'atomic_radius': [1.43, 1.45],
            'formation_energy_per_atom': [-1.2, -0.8],
            'energy_above_hull': [0.02, 0.05],
            'band_gap': [0.0, 0.1],
            'nsites': [2, 2],
            'is_generated': [True, True]
        }
        
        df = pd.DataFrame(test_data)
        
        # Create and train classifier
        ml_classifier = SynthesizabilityClassifier()
        ml_metrics = ml_classifier.train(api_key=None)  # Use synthetic data
        
        print(f"ML classifier trained: {ml_metrics}")
        
        # Test prediction
        results = ml_classifier.predict(df)
        print(f"Prediction results shape: {results.shape}")
        print(f"Prediction columns: {list(results.columns)}")
        
        if 'synthesizability_probability' in results.columns:
            print("ML prediction successful!")
            return True
        else:
            print("ML prediction failed - missing required columns")
            return False
            
    except Exception as e:
        print(f"ML prediction test failed: {e}")
        return False

def test_synthetic_dataset():
    """Test synthetic dataset generation."""
    print("\n=== Testing Synthetic Dataset ===")
    
    try:
        from gradio_app import create_synthetic_dataset
        
        # Create synthetic dataset
        dataset = create_synthetic_dataset(100)
        print(f"Synthetic dataset shape: {dataset.shape}")
        print(f"Synthetic dataset columns: {list(dataset.columns)}")
        
        # Check for required fields
        required_fields = ['formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'nsites']
        missing_fields = [field for field in required_fields if field not in dataset.columns]
        
        if missing_fields:
            print(f"Missing required fields in synthetic dataset: {missing_fields}")
            return False
        else:
            print("All required fields present in synthetic dataset")
            return True
            
    except Exception as e:
        print(f"Synthetic dataset test failed: {e}")
        return False

def test_full_generation_flow():
    """Test the complete generation flow."""
    print("\n=== Testing Full Generation Flow ===")
    
    try:
        from gradio_app import generate_materials, train_vae_model, create_synthetic_dataset
        from sklearn.preprocessing import StandardScaler
        
        # Create synthetic dataset for VAE training
        dataset = create_synthetic_dataset(100)
        
        # Prepare features for VAE
        feature_cols = ['composition_1', 'composition_2', 'formation_energy_per_atom', 'density', 'electronegativity', 'atomic_radius']
        features = dataset[feature_cols].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        print(f"VAE training features shape: {features_scaled.shape}")
        
        # Train VAE
        vae_model = train_vae_model(features_scaled, latent_dim=5, epochs=10)
        print("VAE model trained successfully!")
        
        # Generate materials
        generated_df = generate_materials(vae_model, scaler, num_samples=5)
        
        print(f"Generated materials shape: {generated_df.shape}")
        print(f"Generated materials columns: {list(generated_df.columns)}")
        
        # Check that field mapping was applied
        required_fields = ['formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'nsites', 'density', 'electronegativity', 'atomic_radius']
        missing_fields = [field for field in required_fields if field not in generated_df.columns]
        
        if missing_fields:
            print(f"Missing required fields after generation: {missing_fields}")
            return False
        else:
            print("All required fields present after generation")
            return True
            
    except Exception as e:
        print(f"Full generation flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Complete Solution Test Suite")
    print("=" * 50)
    
    tests = [
        ("API Authentication", test_api_authentication),
        ("Field Mapping", test_field_mapping),
        ("ML Prediction", test_ml_prediction),
        ("Synthetic Dataset", test_synthetic_dataset),
        ("Full Generation Flow", test_full_generation_flow)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            print(f"\n{test_name}: ❌ FAIL - {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The complete solution is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)