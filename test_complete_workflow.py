#!/usr/bin/env python3
"""
Complete workflow test for Docker deployment
Tests the full end-to-end process from initialization to material generation and analysis
"""

import sys
import os
sys.path.append('/app')

def test_complete_workflow():
    """Test the complete workflow from initialization to analysis."""
    print("=" * 80)
    print("🧪 COMPLETE DOCKER WORKFLOW TEST")
    print("=" * 80)
    
    try:
        # Import all required modules
        from synthesizability_predictor import SynthesizabilityClassifier, create_synthetic_dataset_fallback, LLMSynthesizabilityPredictor
        from gradio_app import train_vae_model, generate_materials, run_synthesizability_analysis, calculate_synthesis_priority
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        import numpy as np
        
        print("✅ All imports successful")
        
        # Step 1: Initialize ML Classifier
        print("\n1️⃣ Initializing ML Classifier...")
        ml_classifier = SynthesizabilityClassifier()
        ml_metrics = ml_classifier.train(api_key=None)
        print(f"   ✅ ML Classifier trained: Accuracy={ml_metrics['accuracy']:.3f}, F1={ml_metrics['f1_score']:.3f}")
        
        # Step 2: Create and prepare VAE training data
        print("\n2️⃣ Preparing VAE training data...")
        dataset = create_synthetic_dataset_fallback(1000)
        feature_cols = ['composition_1', 'composition_2', 'formation_energy_per_atom', 'density', 'electronegativity', 'atomic_radius']
        
        # Verify all required fields
        required_fields = ['formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'nsites']
        missing_cols = [col for col in required_fields if col not in dataset.columns]
        if missing_cols:
            raise ValueError(f"Missing required fields: {missing_cols}")
        
        features = dataset[feature_cols].values
        vae_scaler = StandardScaler()
        features_scaled = vae_scaler.fit_transform(features)
        print(f"   ✅ VAE training data prepared: {len(dataset)} materials, {features_scaled.shape[1]} features")
        
        # Step 3: Train VAE model
        print("\n3️⃣ Training VAE model...")
        vae_model = train_vae_model(features_scaled, latent_dim=5, epochs=50)
        print("   ✅ VAE model trained successfully")
        
        # Step 4: Generate new materials
        print("\n4️⃣ Generating new materials...")
        generated_df = generate_materials(vae_model, vae_scaler, num_samples=50)
        print(f"   ✅ Generated {len(generated_df)} new materials")
        
        # Step 5: Run synthesizability analysis
        print("\n5️⃣ Running synthesizability analysis...")
        llm_predictor = LLMSynthesizabilityPredictor()
        analysis_results = run_synthesizability_analysis(generated_df, ml_classifier, llm_predictor)
        print(f"   ✅ Analysis completed: {len(analysis_results)} materials analyzed")
        
        # Step 6: Calculate synthesis priorities
        print("\n6️⃣ Calculating synthesis priorities...")
        priority_results = calculate_synthesis_priority(analysis_results)
        print(f"   ✅ Priority calculation completed")
        
        # Step 7: Verify results quality
        print("\n7️⃣ Verifying results quality...")
        
        # Check prediction distributions
        synthesizable_count = analysis_results['ensemble_prediction'].sum()
        high_confidence = analysis_results[analysis_results['ensemble_confidence'] > 0.8]
        
        print(f"   📊 Synthesizable materials: {synthesizable_count}/{len(analysis_results)} ({synthesizable_count/len(analysis_results)*100:.1f}%)")
        print(f"   📊 High confidence predictions: {len(high_confidence)}/{len(analysis_results)} ({len(high_confidence)/len(analysis_results)*100:.1f}%)")
        
        # Check priority distribution
        high_priority = priority_results[priority_results['synthesis_priority_score'] > 0.7]
        print(f"   📊 High priority materials: {len(high_priority)}/{len(priority_results)} ({len(high_priority)/len(priority_results)*100:.1f}%)")
        
        # Verify required columns are present
        required_analysis_cols = ['ensemble_probability', 'ensemble_confidence', 'energy_above_hull', 'density']
        missing_analysis_cols = [col for col in required_analysis_cols if col not in analysis_results.columns]
        if missing_analysis_cols:
            raise ValueError(f"Missing analysis columns: {missing_analysis_cols}")
        
        required_priority_cols = ['synthesis_priority_score', 'synthesis_priority_rank']
        missing_priority_cols = [col for col in required_priority_cols if col not in priority_results.columns]
        if missing_priority_cols:
            raise ValueError(f"Missing priority columns: {missing_priority_cols}")
        
        print("   ✅ All required columns present in results")
        
        # Step 8: Test cost-benefit analysis
        print("\n8️⃣ Testing cost-benefit analysis...")
        from synthesizability_predictor import cost_benefit_analysis
        
        # Test on top material
        top_material = priority_results.iloc[0].to_dict()
        cba_result = cost_benefit_analysis(top_material)
        
        print(f"   💰 Recommended method: {cba_result['recommended_method']}")
        print(f"   💰 Success probability: {cba_result['success_probability']:.3f}")
        print(f"   💰 Benefit-cost ratio: {cba_result['benefit_cost_ratio']:.2f}")
        
        # Step 9: Test composition analysis
        print("\n9️⃣ Testing composition analysis...")
        from synthesizability_predictor import add_composition_analysis_to_dataframe
        
        composition_results = add_composition_analysis_to_dataframe(generated_df.head(5))
        print(f"   🧪 Composition analysis completed for {len(composition_results)} materials")
        
        # Verify composition columns
        if 'wt%' not in composition_results.columns or 'feedstock_g_per_100g' not in composition_results.columns:
            raise ValueError("Missing composition analysis columns")
        
        print("   ✅ Composition analysis successful")
        
        print("\n" + "=" * 80)
        print("🎉 COMPLETE WORKFLOW TEST PASSED!")
        print("=" * 80)
        print("\n📋 Summary:")
        print(f"   • ML Classifier: ✅ Trained (Accuracy: {ml_metrics['accuracy']:.3f})")
        print(f"   • VAE Model: ✅ Trained and generating materials")
        print(f"   • Material Generation: ✅ {len(generated_df)} materials created")
        print(f"   • Synthesizability Analysis: ✅ Complete with ML + LLM predictions")
        print(f"   • Priority Ranking: ✅ Multi-criteria optimization")
        print(f"   • Cost-Benefit Analysis: ✅ Economic evaluation")
        print(f"   • Composition Analysis: ✅ Weight percentages and feedstock calculations")
        print(f"   • Field Validation: ✅ All required fields present")
        print("\n🚀 Docker deployment is FULLY FUNCTIONAL!")
        print("   Ready for production use with the two-step process:")
        print("   1. Initialize Models (train classifiers and VAE)")
        print("   2. Generate Materials (create candidates and run analysis)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Complete workflow test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complete_workflow()
    sys.exit(0 if success else 1)