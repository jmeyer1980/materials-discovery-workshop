import pandas as pd

from centralized_field_mapping import find_field_name, ensure_required_fields, REQUIRED_FIELDS


def test_find_field_name_returns_matching_variant():
    df_columns = ["bandgap", "mass_density", "foo"]
    match = find_field_name(["band_gap", "bandgap", "gap"], df_columns)
    assert match == "bandgap"


def test_ensure_required_fields_adds_required_columns():
    sample_df = pd.DataFrame(
        {
            "formation_energy": [-1.2, -0.8],
            "bandgap": [0.0, 0.2],
            "num_sites": [4, 6],
            "mass_density": [5.2, 6.1],
            "electronegativity_avg": [1.7, 1.9],
            "atomic_radius_avg": [1.3, 1.4],
        }
    )

    result = ensure_required_fields(sample_df)

    for field in REQUIRED_FIELDS:
        assert field in result.columns

    assert len(result) == len(sample_df)
