#!/usr/bin/env python3
"""
Comprehensive test script to verify the Docker solution works correctly.
This tests all the fixes we implemented for the field mapping and ML prediction issues.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path to import modules
sys.path.insert(0, '/app')

def test_field_mapping():
    """Test the centralized field mapping functionality."""
    print("🧪 Testing Field Mapping...")
    
    try:
        from centralized_field_mapping import apply_field_mapping_to_generation
        
        # Create a test DataFrame with missing fields
        test_df = pd.DataFrame({
            'element_1': ['Al', 'Ti'],
            'element_2': ['Ti', 'Al'],
            'composition_1': [0.5, 0.6],
            'composition_2': [0.5, 0.4],
            'formula': ['Al0.500Ti0.500', 'Ti0.600Al0.400'],
            'melting_point': [1000, 1200],
            'density': [3.5, 4.2],
            'electronegativity': [1.5, 1.6],
            'atomic_radius': [1.43, 1.45],
            'is_generated': [True, True]
        })
        
        print(f"  Original DataFrame columns: {list(test_df.columns)}")
        
        # Apply field mapping
        mapped_df = apply_field_mapping_to_generation(test_df)
        
        print(f"  Mapped DataFrame columns: {list(mapped_df.columns)}")
        
        # Check that all required fields are present
        required_fields = ['formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'nsites', 'density', 'electronegativity', 'atomic_radius']
        missing_fields = [field for field in required_fields if field not in mapped_df.columns]
        
        if missing_fields:
            print(f"  ❌ Missing fields after mapping: {missing_fields}")
            return False
        else:
            print(f"  ✅ All required fields present after mapping")
            return True
            
    except Exception as e:
        print(f"  ❌ Field mapping test failed: {e}")
        return False


def test_ml_prediction():
    """Test ML prediction with field mapping fallback."""
    print("🧪 Testing ML Prediction...")
    
    try:
        from synthesizability_predictor import SynthesizabilityClassifier
        
        # Create a test DataFrame with missing fields (simulating the original issue)
        test_df = pd.DataFrame({
            'element_1': ['Al', 'Ti'],
            'element_2': ['Ti', 'Al'],
            'composition_1': [0.5, 0.6],
            'composition_2': [0.5, 0.4],
            'formula': ['Al0.500Ti0.500', 'Ti0.600Al0.400'],
            'melting_point': [1000, 1200],
            'density': [3.5, 4.2],
            'electronegativity': [1.5, 1.6],
            'atomic_radius': [1.43, 1.45],
            'is_generated': [True, True]
        })
        
        print(f"  Test DataFrame columns: {list(test_df.columns)}")
        
        # Initialize classifier
        classifier = SynthesizabilityClassifier(model_type='random_forest')
        
        # Train the model (this will use synthetic data in Docker)
        print("  Training ML classifier...")
        metrics = classifier.train()
        print(f"  Training completed with accuracy: {metrics['accuracy']:.3f}")
        
        # Test prediction - this should trigger the field mapping fallback
        print("  Testing prediction with missing fields...")
        result_df = classifier.predict(test_df)
        
        print(f"  Result DataFrame columns: {list(result_df.columns)}")
        
        # Check that predictions were made
        if 'synthesizability_prediction' in result_df.columns:
            print(f"  ✅ ML prediction successful")
            print(f"  Predictions: {list(result_df['synthesizability_prediction'])}")
            return True
        else:
            print(f"  ❌ ML prediction failed - no prediction column")
            return False
            
    except Exception as e:
        print(f"  ❌ ML prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthesizability_analysis():
    """Test the complete synthesizability analysis workflow."""
    print("🧪 Testing Synthesizability Analysis...")
    
    try:
        from gradio_app import run_synthesizability_analysis
        from synthesizability_predictor import SynthesizabilityClassifier, LLMSynthesizabilityPredictor
        
        # Create test materials DataFrame
        test_df = pd.DataFrame({
            'element_1': ['Al', 'Ti'],
            'element_2': ['Ti', 'Al'],
            'composition_1': [0.5, 0.6],
            'composition_2': [0.5, 0.4],
            'formula': ['Al0.500Ti0.500', 'Ti0.600Al0.400'],
            'melting_point': [1000, 1200],
            'density': [3.5, 4.2],
            'electronegativity': [1.5, 1.6],
            'atomic_radius': [1.43, 1.45],
            'is_generated': [True, True]
        })
        
        # Initialize classifiers
        ml_classifier = SynthesizabilityClassifier(model_type='random_forest')
        llm_predictor = LLMSynthesizabilityPredictor()
        
        # Train the classifier
        print("  Training ML classifier...")
        metrics = ml_classifier.train()
        print(f"  Training completed with accuracy: {metrics['accuracy']:.3f}")
        
        # Test with a small number of materials
        print("  Running synthesizability analysis...")
        result = run_synthesizability_analysis(test_df, ml_classifier, llm_predictor)
        
        if result is not None:
            print(f"  ✅ Synthesizability analysis completed")
            print(f"  Generated {len(result)} materials")
            
            # Check for key columns
            expected_columns = ['synthesizability_prediction', 'synthesizability_probability', 'synthesizability_confidence']
            missing_columns = [col for col in expected_columns if col not in result.columns]
            
            if missing_columns:
                print(f"  ⚠️  Missing expected columns: {missing_columns}")
                return False
            else:
                print(f"  ✅ All expected columns present")
                return True
        else:
            print(f"  ❌ Synthesizability analysis returned None")
            return False
            
    except Exception as e:
        print(f"  ❌ Synthesizability analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_authentication():
    """Test API authentication handler."""
    print("🧪 Testing API Authentication...")
    
    try:
        from api_authentication_handler import APIAuthenticationHandler
        
        # Test with valid key (but we'll mock the response since we don't have real API access)
        valid_key = "test_key_123"
        os.environ['MP_API_KEY'] = valid_key
        
        # Create handler with test key
        handler = APIAuthenticationHandler(valid_key)
        
        # Test validation - this will fail against real API but we can test the structure
        validation_result = handler.validate_api_key()
        
        # Check that the validation result has the expected structure
        expected_keys = ['valid', 'error', 'message', 'suggestions']
        if all(key in validation_result for key in expected_keys):
            print(f"  ✅ API authentication handler structure is correct")
            print(f"  Validation result: {validation_result['message']}")
            return True
        else:
            print(f"  ❌ API authentication handler missing expected keys")
            print(f"  Expected: {expected_keys}")
            print(f"  Got: {list(validation_result.keys())}")
            return False
        
    except Exception as e:
        print(f"  ❌ API authentication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🚀 Starting Docker Solution Tests")
    print("=" * 50)
    
    tests = [
        test_field_mapping,
        test_ml_prediction,
        test_synthesizability_analysis,
        test_api_authentication
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"  ❌ Test {test.__name__} crashed: {e}")
            results.append(False)
            print()
    
    # Summary
    print("=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Docker solution is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)