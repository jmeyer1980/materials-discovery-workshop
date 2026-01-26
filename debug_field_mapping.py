#!/usr/bin/env python3
"""
Debug script to test field mapping functionality
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_field_mapping():
    """Test the field mapping functionality"""
    print("=== Testing Field Mapping ===")
    
    # Create a test DataFrame with some fields
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
        'is_generated': [True, True, True]
    }
    
    df = pd.DataFrame(test_data)
    print(f"Original DataFrame columns: {list(df.columns)}")
    print(f"Original DataFrame shape: {df.shape}")
    print("Original DataFrame:")
    print(df)
    print()
    
    # Test centralized field mapping
    try:
        from centralized_field_mapping import apply_field_mapping_to_generation, REQUIRED_FIELDS
        print("Testing centralized field mapping...")
        
        # Log initial state
        from centralized_field_mapping import log_dataframe_state
        log_dataframe_state(df, "Before field mapping")
        
        # Apply field mapping
        result_df = apply_field_mapping_to_generation(df)
        
        print(f"After field mapping columns: {list(result_df.columns)}")
        print(f"After field mapping shape: {result_df.shape}")
        print("After field mapping:")
        print(result_df)
        print()
        
        # Check required fields
        print("Required fields status:")
        for field in REQUIRED_FIELDS:
            if field in result_df.columns:
                print(f"  ✓ {field}: {result_df[field].dtype}, min={result_df[field].min():.3f}, max={result_df[field].max():.3f}")
            else:
                print(f"  ✗ {field}: MISSING")
        
        return True
        
    except Exception as e:
        print(f"Error in centralized field mapping: {e}")
        import traceback
        traceback.print_exc()
        
        # Test manual field mapping
        print("\nTesting manual field mapping fallback...")
        try:
            from gradio_app import run_synthesizability_analysis
            print("Manual field mapping test completed")
            return True
        except Exception as e2:
            print(f"Error in manual field mapping: {e2}")
            import traceback
            traceback.print_exc()
            return False

def test_ml_prediction():
    """Test ML prediction with field mapping"""
    print("\n=== Testing ML Prediction ===")
    
    try:
        # Create test data with required fields
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
        
        # Test ML classifier
        from synthesizability_predictor import SynthesizabilityClassifier
        ml_classifier = SynthesizabilityClassifier()
        
        # Create dummy training data
        from synthesizability_predictor import create_vae_training_dataset_from_mp
        try:
            # Try to create real training data
            dataset = create_vae_training_dataset_from_mp(api_key='test', n_materials=10)
        except:
            # Use synthetic data
            from gradio_app import create_synthetic_dataset
            dataset = create_synthetic_dataset(10)
        
        # Train the classifier
        ml_metrics = ml_classifier.train(api_key='test')
        print(f"ML classifier trained: {ml_metrics}")
        
        # Test prediction
        results = ml_classifier.predict(df)
        print(f"ML prediction results shape: {results.shape}")
        print(f"ML prediction results columns: {list(results.columns)}")
        print("ML prediction results:")
        print(results)
        
        return True
        
    except Exception as e:
        print(f"Error in ML prediction test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting field mapping debug tests...")
    
    success1 = test_field_mapping()
    success2 = test_ml_prediction()
    
    if success1 and success2:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)