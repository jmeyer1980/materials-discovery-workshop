import pandas as pd

from synthesizability_predictor import (
    STABILITY_THRESHOLDS,
    add_composition_analysis_to_dataframe,
    calculate_synthesis_priority,
    calculate_thermodynamic_stability_score,
    calculate_weight_percentages,
    classify_stability_category,
    cost_benefit_analysis,
    create_synthetic_dataset_fallback,
    generate_mock_icsd_data,
    generate_mock_mp_only_data,
    generate_precursor_recommendations,
    get_atomic_weight,
    get_stability_info,
    recommend_synthesis_method,
    summarize_tertiary_coverage,
    thermodynamic_stability_check,
)


# --- classify_stability_category ---

def test_classify_stability_returns_highly_stable_at_boundary():
    threshold = STABILITY_THRESHOLDS["highly_stable"]["energy_above_hull_max"]
    assert classify_stability_category(threshold) == "highly_stable"


def test_classify_stability_below_boundary_is_highly_stable():
    assert classify_stability_category(0.01) == "highly_stable"


def test_classify_stability_returns_marginal_between_thresholds():
    low = STABILITY_THRESHOLDS["highly_stable"]["energy_above_hull_max"]
    high = STABILITY_THRESHOLDS["marginal"]["energy_above_hull_max"]
    mid = (low + high) / 2
    assert classify_stability_category(mid) == "marginal"


def test_classify_stability_returns_risky_above_marginal():
    assert classify_stability_category(0.5) == "risky"


def test_classify_stability_returns_risky_for_large_value():
    assert classify_stability_category(10.0) == "risky"


# --- get_stability_info ---

def test_get_stability_info_highly_stable_structure():
    info = get_stability_info(0.01)
    assert info["category"] == "highly_stable"
    assert info["synthesis_confidence"] >= 0.8
    assert "description" in info
    assert info["energy_above_hull"] == 0.01


def test_get_stability_info_marginal_structure():
    info = get_stability_info(0.07)
    assert info["category"] == "marginal"
    assert 0.0 < info["synthesis_confidence"] < 0.9


def test_get_stability_info_risky_has_low_confidence():
    info = get_stability_info(1.0)
    assert info["category"] == "risky"
    assert info["synthesis_confidence"] < 0.5


def test_get_stability_info_includes_threshold_used():
    info = get_stability_info(0.02)
    assert "threshold_used" in info


# --- get_atomic_weight ---

def test_get_atomic_weight_returns_positive_for_al():
    weight = get_atomic_weight("Al")
    assert weight > 20.0


def test_get_atomic_weight_returns_positive_for_ti():
    weight = get_atomic_weight("Ti")
    assert weight > 40.0


def test_get_atomic_weight_fallback_for_unknown_element():
    weight = get_atomic_weight("Xx")
    assert weight >= 0.0


# --- calculate_weight_percentages ---

def test_calculate_weight_percentages_binary_sums_to_100():
    result = calculate_weight_percentages(
        {
            "element_1": "Al",
            "element_2": "Ti",
            "composition_1": 0.5,
            "composition_2": 0.5,
        }
    )
    assert abs(result["total_weight_percent"] - 100.0) < 1e-3
    assert "Al" in result["wt%"] and "Ti" in result["wt%"]


def test_calculate_weight_percentages_ternary_sums_to_100():
    result = calculate_weight_percentages(
        {
            "element_1": "Fe",
            "element_2": "Ni",
            "element_3": "Cr",
            "composition_1": 0.6,
            "composition_2": 0.3,
            "composition_3": 0.1,
        }
    )
    assert abs(result["total_weight_percent"] - 100.0) < 1e-3


def test_calculate_weight_percentages_no_elements_returns_zero():
    result = calculate_weight_percentages({})
    assert result["total_weight_percent"] == 0.0


def test_calculate_weight_percentages_feedstock_string_populated():
    result = calculate_weight_percentages(
        {
            "element_1": "Cu",
            "element_2": "Zn",
            "composition_1": 0.7,
            "composition_2": 0.3,
        }
    )
    feedstock = result["feedstock_g_per_100g"]
    assert isinstance(feedstock, str)
    assert "Cu" in feedstock or "Zn" in feedstock


# --- add_composition_analysis_to_dataframe ---

def test_add_composition_analysis_adds_columns():
    df = pd.DataFrame(
        [
            {
                "element_1": "Al",
                "element_2": "Ti",
                "composition_1": 0.6,
                "composition_2": 0.4,
            }
        ]
    )
    result = add_composition_analysis_to_dataframe(df)
    assert "wt%" in result.columns
    assert "feedstock_g_per_100g" in result.columns


def test_add_composition_analysis_preserves_row_count():
    df = pd.DataFrame(
        [
            {"element_1": "Al", "element_2": "Ti", "composition_1": 0.5, "composition_2": 0.5},
            {"element_1": "Fe", "element_2": "Ni", "composition_1": 0.4, "composition_2": 0.6},
        ]
    )
    result = add_composition_analysis_to_dataframe(df)
    assert len(result) == 2


# --- summarize_tertiary_coverage ---

def test_summarize_tertiary_coverage_binary_only():
    df = pd.DataFrame({"composition_3": [0.0, 0.0, 0.0]})
    result = summarize_tertiary_coverage(df)
    assert result["tertiary_rows"] == 0
    assert result["recommended"] is False


def test_summarize_tertiary_coverage_all_ternary():
    df = pd.DataFrame({"composition_3": [0.2, 0.3, 0.1] * 20})
    result = summarize_tertiary_coverage(df)
    assert result["tertiary_rows"] == 60
    assert 0.0 < result["tertiary_fraction"] <= 1.0


def test_summarize_tertiary_coverage_empty_df():
    result = summarize_tertiary_coverage(pd.DataFrame())
    assert result["total_rows"] == 0
    assert result["recommended"] is False


def test_summarize_tertiary_coverage_missing_composition3_column():
    df = pd.DataFrame({"composition_1": [0.5, 0.5], "composition_2": [0.5, 0.5]})
    result = summarize_tertiary_coverage(df)
    assert result["tertiary_rows"] == 0


# --- thermodynamic_stability_check ---

def test_thermodynamic_stability_check_adds_expected_columns():
    df = pd.DataFrame(
        {
            "energy_above_hull": [0.02, 0.08, 0.3],
            "formation_energy_per_atom": [-2.0, -1.0, -0.5],
        }
    )
    result = thermodynamic_stability_check(df)
    assert "thermodynamically_stable" in result.columns
    assert "formation_energy_reasonable" in result.columns
    assert "overall_stability" in result.columns


def test_thermodynamic_stability_marks_stable_correctly():
    df = pd.DataFrame(
        {
            "energy_above_hull": [0.05],
            "formation_energy_per_atom": [-2.0],
        }
    )
    result = thermodynamic_stability_check(df)
    assert bool(result["thermodynamically_stable"].iloc[0]) is True
    assert bool(result["overall_stability"].iloc[0]) is True


def test_thermodynamic_stability_marks_unstable_correctly():
    df = pd.DataFrame(
        {
            "energy_above_hull": [0.5],
            "formation_energy_per_atom": [-2.0],
        }
    )
    result = thermodynamic_stability_check(df)
    assert bool(result["thermodynamically_stable"].iloc[0]) is False


# --- generate_mock_icsd_data ---

def test_generate_mock_icsd_data_shape_and_columns():
    df = generate_mock_icsd_data(n_samples=50)
    assert len(df) == 50
    for col in [
        "composition_1", "composition_2", "composition_3",
        "formation_energy_per_atom", "band_gap", "energy_above_hull",
        "electronegativity", "atomic_radius", "nsites", "density", "synthesizable",
    ]:
        assert col in df.columns


def test_generate_mock_icsd_data_all_labeled_synthesizable():
    df = generate_mock_icsd_data(n_samples=30)
    assert (df["synthesizable"] == 1).all()


def test_generate_mock_icsd_data_compositions_in_range():
    df = generate_mock_icsd_data(n_samples=40)
    assert (df["composition_1"] >= 0).all()
    assert (df["composition_2"] >= 0).all()
    assert (df["composition_3"] >= 0).all()


def test_generate_mock_icsd_data_energy_hull_bounded():
    df = generate_mock_icsd_data(n_samples=100)
    assert (df["energy_above_hull"] >= 0).all()
    assert (df["energy_above_hull"] <= 0.5).all()


# --- generate_mock_mp_only_data ---

def test_generate_mock_mp_only_data_shape_and_columns():
    df = generate_mock_mp_only_data(n_samples=50)
    assert len(df) == 50
    for col in [
        "composition_1", "composition_2", "composition_3",
        "formation_energy_per_atom", "band_gap", "energy_above_hull",
        "electronegativity", "atomic_radius", "nsites", "density", "synthesizable",
    ]:
        assert col in df.columns


def test_generate_mock_mp_only_data_all_labeled_not_synthesizable():
    df = generate_mock_mp_only_data(n_samples=30)
    assert (df["synthesizable"] == 0).all()


def test_generate_mock_mp_only_data_has_broader_energy_distribution():
    icsd = generate_mock_icsd_data(n_samples=200)
    mp = generate_mock_mp_only_data(n_samples=200)
    # MP-only data should have higher mean energy above hull than ICSD data
    assert mp["energy_above_hull"].mean() > icsd["energy_above_hull"].mean()


def test_thermodynamic_stability_check_preserves_row_count():
    df = pd.DataFrame(
        {
            "energy_above_hull": [0.0, 0.05, 0.2, 0.5],
            "formation_energy_per_atom": [-3.0, -1.5, 0.5, -0.2],
        }
    )
    result = thermodynamic_stability_check(df)
    assert len(result) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Wave-3: calculate_thermodynamic_stability_score
# ─────────────────────────────────────────────────────────────────────────────

def test_thermo_stability_score_highly_stable():
    result = calculate_thermodynamic_stability_score({"energy_above_hull": 0.01})
    assert result["thermodynamic_stability_category"] == "highly_stable"
    assert result["thermodynamic_stability_score"] >= 0.8


def test_thermo_stability_score_marginal():
    result = calculate_thermodynamic_stability_score({"energy_above_hull": 0.05})
    assert result["thermodynamic_stability_category"] == "marginal"


def test_thermo_stability_score_unstable():
    result = calculate_thermodynamic_stability_score({"energy_above_hull": 0.5})
    assert result["thermodynamic_stability_category"] == "unstable"
    assert result["stability_color"] == "red"


def test_thermo_stability_score_semiconductor_boost():
    """Band gap in [0.5, 4.0] boosts the score."""
    without = calculate_thermodynamic_stability_score({"energy_above_hull": 0.05, "band_gap": 0.0})
    with_bg = calculate_thermodynamic_stability_score({"energy_above_hull": 0.05, "band_gap": 2.0})
    assert with_bg["thermodynamic_stability_score"] > without["thermodynamic_stability_score"]


def test_thermo_stability_score_penalty_extreme_formation_energy():
    """Formation energy < -10 should reduce the score."""
    normal = calculate_thermodynamic_stability_score({"energy_above_hull": 0.01, "formation_energy_per_atom": -1.0})
    extreme = calculate_thermodynamic_stability_score({"energy_above_hull": 0.01, "formation_energy_per_atom": -15.0})
    assert extreme["thermodynamic_stability_score"] < normal["thermodynamic_stability_score"]


def test_thermo_stability_score_returns_all_keys():
    result = calculate_thermodynamic_stability_score({"energy_above_hull": 0.02})
    for key in ("thermodynamic_stability_score", "thermodynamic_stability_category", "stability_color",
                "energy_above_hull", "formation_energy_per_atom", "density", "band_gap"):
        assert key in result


# ─────────────────────────────────────────────────────────────────────────────
# create_synthetic_dataset_fallback
# ─────────────────────────────────────────────────────────────────────────────

def test_create_synthetic_dataset_fallback_row_count():
    df = create_synthetic_dataset_fallback(n_samples=50)
    assert len(df) == 50


def test_create_synthetic_dataset_fallback_required_columns():
    df = create_synthetic_dataset_fallback(n_samples=20)
    for col in ("element_1", "element_2", "composition_1", "composition_2",
                "melting_point", "density", "energy_above_hull"):
        assert col in df.columns


def test_create_synthetic_dataset_fallback_melting_point_positive():
    df = create_synthetic_dataset_fallback(n_samples=100)
    assert (df["melting_point"] > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# generate_precursor_recommendations
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_precursor_recommendations_always_includes_cvd():
    """CVD is always appended as a fallback option."""
    result = generate_precursor_recommendations({"energy_above_hull": 0.0, "band_gap": 5.0})
    methods = [r["method"] for r in result]
    assert "Chemical Vapor Deposition" in methods


def test_generate_precursor_recommendations_arc_melting_for_low_bandgap():
    """Arc Melting should appear for low-band-gap (metallic) material."""
    result = generate_precursor_recommendations({"energy_above_hull": 0.1, "band_gap": 0.5})
    methods = [r["method"] for r in result]
    assert "Arc Melting" in methods


def test_generate_precursor_recommendations_solid_state_for_stable():
    """Solid State Reaction appears for highly stable materials (stability_score > 0.7)."""
    # energy_above_hull = 0 → stability_score = 1 - 0/0.5 = 1.0 > 0.7
    result = generate_precursor_recommendations({"energy_above_hull": 0.0, "band_gap": 5.0})
    methods = [r["method"] for r in result]
    assert "Solid State Reaction" in methods


# ─────────────────────────────────────────────────────────────────────────────
# calculate_synthesis_priority
# ─────────────────────────────────────────────────────────────────────────────

def test_calculate_synthesis_priority_adds_score_and_rank_columns():
    df = pd.DataFrame({
        "energy_above_hull": [0.01, 0.05, 0.3],
        "nsites": [4, 8, 16],
    })
    result = calculate_synthesis_priority(df)
    assert "synthesis_priority_score" in result.columns
    assert "synthesis_priority_rank" in result.columns


def test_calculate_synthesis_priority_lower_hull_gets_higher_score():
    df = pd.DataFrame({
           "energy_above_hull": [0.01, 0.8],
        "nsites": [4, 4],
    })
    result = calculate_synthesis_priority(df)
    assert result["synthesis_priority_score"].iloc[0] > result["synthesis_priority_score"].iloc[1]


# ─────────────────────────────────────────────────────────────────────────────
# recommend_synthesis_method
# ─────────────────────────────────────────────────────────────────────────────

def _all_equipment():
    from synthesizability_predictor import EQUIPMENT_CATEGORIES
    equipment = []
    for items in EQUIPMENT_CATEGORIES.values():
        equipment.extend(items)
    return equipment


def test_recommend_synthesis_method_returns_method_with_all_equipment():
    result = recommend_synthesis_method(
        {"energy_above_hull": 0.02, "band_gap": 0.5, "density": 7.0},
        available_equipment=_all_equipment(),
    )
    assert result["method_key"] is not None
    assert "method_name" in result


def test_recommend_synthesis_method_no_equipment_returns_none():
    result = recommend_synthesis_method(
        {"energy_above_hull": 0.02, "band_gap": 0.5, "density": 7.0},
        available_equipment=[],
    )
    assert result["method_key"] is None


# ─────────────────────────────────────────────────────────────────────────────
# cost_benefit_analysis
# ─────────────────────────────────────────────────────────────────────────────

def test_cost_benefit_analysis_no_equipment_returns_zero_cost():
    result = cost_benefit_analysis(
        {"energy_above_hull": 0.02, "band_gap": 1.0, "density": 7.0},
        available_equipment=[],
    )
    assert result["total_cost"] == 0
    assert result["benefit_cost_ratio"] == 0


def test_cost_benefit_analysis_with_all_equipment_has_positive_cost():
    result = cost_benefit_analysis(
        {
            "energy_above_hull": 0.02,
            "band_gap": 1.0,
            "density": 7.0,
            "thermodynamic_stability_score": 0.8,
            "ensemble_probability": 0.8,
        },
        available_equipment=_all_equipment(),
    )
    assert result["total_cost"] > 0
    assert result["recommended_method"] != "No suitable method available"
