import pandas as pd

from field_mapping_utils import (
    REQUIRED_FIELDS_ML_CLASSIFIER,
    standardize_dataframe,
)
from synthesizability_predictor import (
    STABILITY_THRESHOLDS,
    calculate_thermodynamic_stability_score,
)


def test_converted_diagnostic_field_standardization_handles_inconsistent_names():
    bad_data = pd.DataFrame(
        {
            "comp_1": [0.5, 0.6],
            "comp_2": [0.5, 0.4],
            "density": [7.8, 8.2],
            "electroneg": [1.8, 1.9],
            "atomic_rad": [1.4, 1.5],
            "band_gap": [0.0, 1.2],
            "energy_hull": [0.02, 0.08],
            "formation_energy": [-2.5, -1.8],
            "sites": [4, 6],
        }
    )

    standardized = standardize_dataframe(
        bad_data,
        required_fields=REQUIRED_FIELDS_ML_CLASSIFIER,
        context="tests.converted_diagnostic.standardization",
    )

    for field in REQUIRED_FIELDS_ML_CLASSIFIER:
        assert field in standardized.columns

    assert len(standardized) == 2


def test_converted_diagnostic_thermodynamic_stability_scoring_rules():
    highly_stable = calculate_thermodynamic_stability_score(
        {
            "energy_above_hull": STABILITY_THRESHOLDS["highly_stable"]["energy_above_hull_max"],
            "formation_energy_per_atom": -2.0,
            "density": 8.0,
            "band_gap": 1.2,
        }
    )

    unstable_penalized = calculate_thermodynamic_stability_score(
        {
            "energy_above_hull": 0.3,
            "formation_energy_per_atom": -12.0,
            "density": 40.0,
            "band_gap": 0.0,
        }
    )

    assert highly_stable["thermodynamic_stability_category"] == "highly_stable"
    assert unstable_penalized["thermodynamic_stability_category"] == "unstable"
    assert highly_stable["thermodynamic_stability_score"] > unstable_penalized["thermodynamic_stability_score"]
