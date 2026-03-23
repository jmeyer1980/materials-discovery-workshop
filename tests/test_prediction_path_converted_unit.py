from synthesizability_predictor import LLMSynthesizabilityPredictor


def test_converted_prediction_path_llm_rule_model_predicts_stable_material():
    predictor = LLMSynthesizabilityPredictor()

    stable_material = {
        "energy_above_hull": 0.02,
        "formation_energy_per_atom": -2.0,
        "nsites": 8,
        "density": 6.5,
        "band_gap": 1.6,
        "electronegativity": 1.7,
        "atomic_radius": 1.3,
    }

    result = predictor.predict_synthesizability(stable_material)

    assert result["prediction"] == 1
    assert 0.0 <= result["probability"] <= 1.0
    assert result["score"] > 5.0
    assert any("stability" in reason.lower() for reason in result["reasons"])


def test_converted_prediction_path_llm_rule_model_penalizes_unstable_material():
    predictor = LLMSynthesizabilityPredictor()

    unstable_material = {
        "energy_above_hull": 0.5,
        "formation_energy_per_atom": -12.0,
        "nsites": 50,
        "density": 45.0,
        "band_gap": 12.0,
        "electronegativity": 3.5,
        "atomic_radius": 3.2,
    }

    result = predictor.predict_synthesizability(unstable_material)

    assert result["prediction"] == 0
    assert 0.0 <= result["probability"] <= 1.0
    assert result["score"] < 5.0
    assert any("instability" in reason.lower() for reason in result["reasons"])
