#!/usr/bin/env python3
"""
Test script to verify field mapping fixes work correctly across different environments.

This script tests:
1. Field validation and standardization
2. ML classifier with standardized fields
3. Synthesizability prediction with consistent field names
4. Error handling for missing fields
5. Cross-module field consistency
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to Python path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from field_mapping_utils import (
        validate_dataframe_consistency,
        standardize_dataframe,
        REQUIRED_FIELDS_ML_CLASSIFIER
    )
    from materials_discovery_api import create_ml_features_from_mp_data
    from synthesizability_predictor import SynthesizabilityClassifier, calculate_thermodynamic_stability_score
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_field_validation():
    """Test field validation and standardization."""
    print("\n" + "="*60)
    print("TESTING FIELD VALIDATION AND STANDARDIZATION")
    print("="*60)
    
    # Create test data with inconsistent field names
    test_data = pd.DataFrame({
        'composition_1': [0.5, 0.6, 0.7],
        'composition_2': [0.5, 0.4, 0.3],
        'composition_3': [0.0, 0.0, 0.0],
        'density': [7.8, 8.2, 6.5],
        'electronegativity': [1.8, 1.9, 1.7],  # Correct name
        'atomic_radius': [1.4, 1.5, 1.3],      # Correct name
        'band_gap': [0.0, 1.2, 2.5],
        'energy_above_hull': [0.02, 0.08, 0.15],
        'formation_energy_per_atom': [-2.5, -1.8, -0.5],
        'nsites': [4, 6, 8]
    })
    
    print("Original data columns:", list(test_data.columns))
    print("Required fields:", REQUIRED_FIELDS_ML_CLASSIFIER)
    
    # Test field consistency validation
    try:
        is_consistent = validate_dataframe_consistency(test_data)
        print(f"✅ Field consistency check: {is_consistent}")
    except Exception as e:
        print(f"❌ Field consistency check failed: {e}")
    
    # Test standardization
    try:
        standardized = standardize_dataframe(test_data, REQUIRED_FIELDS_ML_CLASSIFIER, "test_validation")
        print("✅ Standardization successful")
        print("Standardized columns:", list(standardized.columns))
        
        # Check that all required fields are present
        missing_fields = [field for field in REQUIRED_FIELDS_ML_CLASSIFIER if field not in standardized.columns]
        if missing_fields:
            print(f"❌ Missing required fields after standardization: {missing_fields}")
        else:
            print("✅ All required fields present after standardization")
            
    except Exception as e:
        print(f"❌ Standardization failed: {e}")
    
    # Test with data that has wrong field names
    print("\nTesting with inconsistent field names...")
    bad_data = pd.DataFrame({
        'comp_1': [0.5, 0.6],  # Wrong name
        'comp_2': [0.5, 0.4],  # Wrong name
        'density': [7.8, 8.2],
        'electroneg': [1.8, 1.9],  # Wrong name
        'atomic_rad': [1.4, 1.5],  # Wrong name
        'band_gap': [0.0, 1.2],
        'energy_hull': [0.02, 0.08],  # Wrong name
        'formation_energy': [-2.5, -1.8],  # Wrong name
        'sites': [4, 6]  # Wrong name
    })
    
    print("Bad data columns:", list(bad_data.columns))
    
    try:
        standardized_bad = standardize_dataframe(bad_data, REQUIRED_FIELDS_ML_CLASSIFIER, "test_bad_data")
        print("✅ Standardization of bad data successful")
        print("Standardized bad data columns:", list(standardized_bad.columns))
        
        # Check that field names were corrected
        expected_fields = ['composition_1', 'composition_2', 'composition_3', 'density', 
                          'electronegativity', 'atomic_radius', 'band_gap', 
                          'energy_above_hull', 'formation_energy_per_atom', 'nsites']
        
        for field in expected_fields:
            if field in standardized_bad.columns:
                print(f"✅ Field '{field}' correctly mapped")
            else:
                print(f"❌ Field '{field}' missing after standardization")
                
    except Exception as e:
        print(f"❌ Standardization of bad data failed: {e}")


def test_ml_classifier_with_standardized_data():
    """Test ML classifier with standardized field names."""
    print("\n" + "="*60)
    print("TESTING ML CLASSIFIER WITH STANDARDIZED DATA")
    print("="*60)
    
    # Create test data with standardized field names
    test_data = pd.DataFrame({
        'composition_1': [0.5, 0.6, 0.7, 0.4, 0.8],
        'composition_2': [0.5, 0.4, 0.3, 0.6, 0.2],
        'composition_3': [0.0, 0.0, 0.0, 0.0, 0.0],
        'density': [7.8, 8.2, 6.5, 9.1, 5.3],
        'electronegativity': [1.8, 1.9, 1.7, 2.0, 1.5],
        'atomic_radius': [1.4, 1.5, 1.3, 1.6, 1.2],
        'band_gap': [0.0, 1.2, 2.5, 0.8, 3.2],
        'energy_above_hull': [0.02, 0.08, 0.15, 0.05, 0.25],
        'formation_energy_per_atom': [-2.5, -1.8, -0.5, -2.1, -0.8],
        'nsites': [4, 6, 8, 10, 12]
    })
    
    print("Test data shape:", test_data.shape)
    print("Test data columns:", list(test_data.columns))
    
    # Initialize and train classifier
    try:
        classifier = SynthesizabilityClassifier(model_type='random_forest')
        
        # Mock training data with correct field names
        training_data = test_data.copy()
        training_data['synthesizable'] = [1, 1, 0, 1, 0]  # Mock labels
        
        # Train the classifier
        metrics = classifier.train(test_size=0.2, api_key=None)  # Use synthetic data
        print("✅ ML classifier trained successfully")
        print(f"Training metrics: {metrics}")
        
        # Test prediction
        predictions = classifier.predict(test_data)
        print("✅ ML classifier prediction successful")
        print(f"Prediction shape: {predictions.shape}")
        print(f"Prediction columns: {list(predictions.columns)}")
        
        # Check that prediction columns are present
        required_pred_cols = ['synthesizability_prediction', 'synthesizability_probability']
        for col in required_pred_cols:
            if col in predictions.columns:
                print(f"✅ Prediction column '{col}' present")
            else:
                print(f"❌ Prediction column '{col}' missing")
                
    except Exception as e:
        print(f"❌ ML classifier test failed: {e}")
        import traceback
        traceback.print_exc()


def test_synthesizability_prediction():
    """Test synthesizability prediction with consistent field names."""
    print("\n" + "="*60)
    print("TESTING SYNTHESIZABILITY PREDICTION")
    print("="*60)
    
    # Test single material prediction
    material_data = {
        'formation_energy_per_atom': -2.5,
        'band_gap': 1.2,
        'energy_above_hull': 0.02,
        'electronegativity': 1.8,
        'atomic_radius': 1.4,
        'nsites': 8,
        'density': 7.2
    }
    
    print("Test material data:", material_data)
    
    try:
        # Test thermodynamic stability calculation
        stability_info = calculate_thermodynamic_stability_score(material_data)
        print("✅ Thermodynamic stability calculation successful")
        print(f"Stability score: {stability_info['thermodynamic_stability_score']}")
        print(f"Stability category: {stability_info['thermodynamic_stability_category']}")
        
        # Test with DataFrame
        material_df = pd.DataFrame([material_data])
        print("Material DataFrame columns:", list(material_df.columns))
        
        # Test field standardization on single material
        standardized_material = standardize_dataframe(material_df, REQUIRED_FIELDS_ML_CLASSIFIER, "test_single_material")
        print("✅ Single material standardization successful")
        print("Standardized columns:", list(standardized_material.columns))
        
    except Exception as e:
        print(f"❌ Synthesizability prediction test failed: {e}")
        import traceback
        traceback.print_exc()


def test_error_handling():
    """Test error handling for missing or incorrect fields."""
    print("\n" + "="*60)
    print("TESTING ERROR HANDLING")
    print("="*60)
    
    # Test with missing required fields
    incomplete_data = pd.DataFrame({
        'composition_1': [0.5, 0.6],
        'density': [7.8, 8.2],
        # Missing: electronegativity, atomic_radius, band_gap, etc.
    })
    
    print("Incomplete data columns:", list(incomplete_data.columns))
    
    try:
        # Test standardization with missing fields
        standardized = standardize_dataframe(incomplete_data, REQUIRED_FIELDS_ML_CLASSIFIER, "test_incomplete")
        print("✅ Standardization with missing fields handled")
        print("Standardized columns:", list(standardized.columns))
        
        # Check that missing fields were added with default values
        for field in REQUIRED_FIELDS_ML_CLASSIFIER:
            if field in standardized.columns:
                print(f"✅ Field '{field}' present (value: {standardized[field].iloc[0]})")
            else:
                print(f"❌ Field '{field}' missing")
                
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()


def test_cross_module_consistency():
    """Test that field names are consistent across all modules."""
    print("\n" + "="*60)
    print("TESTING CROSS-MODULE FIELD CONSISTENCY")
    print("="*60)
    
    # Create data that mimics what would come from Materials Project API
    mp_like_data = pd.DataFrame({
        'material_id': ['mp-123', 'mp-456'],
        'formula': ['AlTi', 'FeNi'],
        'elements': [['Al', 'Ti'], ['Fe', 'Ni']],
        'nsites': [4, 6],
        'volume': [100.0, 120.0],
        'density': [7.8, 8.5],
        'band_gap': [0.0, 1.2],
        'energy_above_hull': [0.02, 0.08],
        'formation_energy_per_atom': [-2.5, -1.8],
        'total_magnetization': [0.0, 2.5]
    })
    
    print("MP-like data columns:", list(mp_like_data.columns))
    
    try:
        # Test ML features creation (this should handle field mapping)
        ml_features = create_ml_features_from_mp_data(mp_like_data)
        print("✅ ML features creation successful")
        print("ML features columns:", list(ml_features.columns))
        
        # Check that ML features have the correct field names
        for field in REQUIRED_FIELDS_ML_CLASSIFIER:
            if field in ml_features.columns:
                print(f"✅ ML feature field '{field}' present")
            else:
                print(f"❌ ML feature field '{field}' missing")
                
    except Exception as e:
        print(f"❌ Cross-module consistency test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("🧪 STARTING FIELD MAPPING FIXES TEST SUITE")
    print("="*60)
    
    # Run all tests
    test_field_validation()
    test_ml_classifier_with_standardized_data()
    test_synthesizability_prediction()
    test_error_handling()
    test_cross_module_consistency()
    
    print("\n" + "="*60)
    print("🏁 TEST SUITE COMPLETED")
    print("="*60)
    print("If all tests passed, the field mapping fixes should work correctly")
    print("in different environments (local, Docker, cloud deployment).")


if __name__ == "__main__":
    main()