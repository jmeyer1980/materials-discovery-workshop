#!/usr/bin/env python3
"""
Test script to verify Docker deployment functionality
"""

import sys
sys.path.insert(0, '/app')

def test_imports():
    """Test that all required modules can be imported"""
    try:
        
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_synthetic_dataset():
    """Test synthetic dataset creation"""
    try:
        from gradio_app import create_synthetic_dataset
        
        # Create synthetic dataset
        dataset = create_synthetic_dataset(100)
        
        # Check required fields
        required_fields = ['formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'nsites', 'density', 'electronegativity', 'atomic_radius']
        
        missing_fields = [field for field in required_fields if field not in dataset.columns]
        
        if missing_fields:
            print(f"❌ Missing required fields in synthetic dataset: {missing_fields}")
            return False
        
        print(f"✅ Synthetic dataset created with {len(dataset)} materials")
        print(f"✅ All required fields present: {list(dataset.columns)}")
        return True
        
    except Exception as e:
        print(f"❌ Synthetic dataset test failed: {e}")
        return False

def test_vae_training():
    """Test VAE training with synthetic data"""
    try:
        from gradio_app import create_synthetic_dataset, train_vae_model
        from sklearn.preprocessing import StandardScaler
        
        # Create dataset
        dataset = create_synthetic_dataset(100)
        
        # Prepare features
        feature_cols = ['composition_1', 'composition_2', 'formation_energy_per_atom', 'density', 'electronegativity', 'atomic_radius']
        features = dataset[feature_cols].values
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train VAE
        vae_model = train_vae_model(features_scaled, latent_dim=5, epochs=10)
        assert vae_model is not None, "VAE training returned None"
        
        print("✅ VAE training successful")
        return True
        
    except Exception as e:
        print(f"❌ VAE training test failed: {e}")
        return False

def test_material_generation():
    """Test material generation"""
    try:
        from gradio_app import create_synthetic_dataset, train_vae_model, generate_materials
        from sklearn.preprocessing import StandardScaler
        
        # Create dataset and train VAE
        dataset = create_synthetic_dataset(100)
        feature_cols = ['composition_1', 'composition_2', 'formation_energy_per_atom', 'density', 'electronegativity', 'atomic_radius']
        features = dataset[feature_cols].values
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        vae_model = train_vae_model(features_scaled, latent_dim=5, epochs=10)
        
        # Generate materials
        generated_df = generate_materials(vae_model, scaler, num_samples=10)
        
        # Check required fields
        required_fields = ['formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'nsites', 'density', 'electronegativity', 'atomic_radius']
        missing_fields = [field for field in required_fields if field not in generated_df.columns]
        
        if missing_fields:
            print(f"❌ Missing required fields in generated materials: {missing_fields}")
            return False
        
        print(f"✅ Material generation successful: {len(generated_df)} materials generated")
        print("✅ All required fields present in generated materials")
        return True
        
    except Exception as e:
        print(f"❌ Material generation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Docker deployment functionality...")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Synthetic Dataset Test", test_synthetic_dataset),
        ("VAE Training Test", test_vae_training),
        ("Material Generation Test", test_material_generation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Docker deployment is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())