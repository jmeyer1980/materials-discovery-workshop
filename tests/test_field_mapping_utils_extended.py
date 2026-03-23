import numpy as np
import pandas as pd

from field_mapping_utils import (
    REQUIRED_FIELDS_ML_CLASSIFIER,
    REQUIRED_FIELDS_VAE,
    FieldMapper,
    standardize_dataframe,
    validate_dataframe_consistency,
)


def _make_canonical_df(n=3):
    return pd.DataFrame(
        {
            "formation_energy_per_atom": [-1.0, -2.0, -0.5][:n],
            "energy_above_hull": [0.01, 0.05, 0.15][:n],
            "band_gap": [0.0, 1.2, 4.5][:n],
            "nsites": [4, 6, 8][:n],
            "density": [7.8, 8.2, 5.0][:n],
            "electronegativity": [1.8, 1.9, 1.7][:n],
            "atomic_radius": [1.3, 1.4, 1.2][:n],
            "composition_1": [0.5, 0.6, 0.4][:n],
            "composition_2": [0.4, 0.3, 0.5][:n],
            "composition_3": [0.1, 0.1, 0.1][:n],
            "element_1": ["Al", "Ti", "Fe"][:n],
            "element_2": ["Ti", "Ni", "Ni"][:n],
            "formula": ["Al0.5Ti0.5", "Ti0.6Ni0.4", "Fe0.4Ni0.6"][:n],
        }
    )


def test_find_canonical_field_returns_first_match():
    cols = ["bandgap", "density", "nsites"]
    match = FieldMapper.find_canonical_field(["band_gap", "bandgap"], cols)
    assert match == "bandgap"


def test_find_canonical_field_returns_none_when_no_match():
    cols = ["density", "nsites"]
    result = FieldMapper.find_canonical_field(["band_gap", "bandgap"], cols)
    assert result is None


def test_map_fields_to_canonical_renames_variations():
    df = pd.DataFrame(
        {
            "bandgap": [1.0, 2.0],
            "e_above_hull": [0.01, 0.05],
            "formation_energy": [-1.0, -2.0],
        }
    )
    result = FieldMapper.map_fields_to_canonical(df)
    assert "band_gap" in result.columns
    assert "energy_above_hull" in result.columns
    assert "formation_energy_per_atom" in result.columns


def test_map_fields_to_canonical_leaves_canonical_names_unchanged():
    df = _make_canonical_df()
    result = FieldMapper.map_fields_to_canonical(df)
    assert "formation_energy_per_atom" in result.columns
    assert "band_gap" in result.columns


def test_ensure_field_completeness_creates_missing_columns():
    df = pd.DataFrame(
        {
            "bandgap": [1.0, 2.0],
            "density": [7.8, 8.2],
        }
    )
    result = FieldMapper.ensure_field_completeness(df, REQUIRED_FIELDS_ML_CLASSIFIER)
    for field in REQUIRED_FIELDS_ML_CLASSIFIER:
        assert field in result.columns
    assert len(result) == 2


def test_ensure_field_completeness_vae_subset():
    df = pd.DataFrame(
        {
            "comp_1": [0.5, 0.6],
            "comp_2": [0.5, 0.4],
        }
    )
    result = FieldMapper.ensure_field_completeness(df, REQUIRED_FIELDS_VAE)
    for field in REQUIRED_FIELDS_VAE:
        assert field in result.columns


def test_ensure_required_fields_renames_known_variation():
    """Exercises the find-and-rename branch of FieldMapper.ensure_required_fields."""
    df = pd.DataFrame({"bandgap": [1.0, 2.0], "density": [7.8, 8.2]})
    result = FieldMapper.ensure_required_fields(df, ["band_gap"])
    assert "band_gap" in result.columns


def test_ensure_required_fields_no_variation_match_leaves_df_unchanged():
    """Exercises the 'Could not find' branch."""
    df = pd.DataFrame({"density": [7.8, 8.2]})
    result = FieldMapper.ensure_required_fields(df, ["band_gap"])
    # band_gap variations not present, so column not added (only rename attempted)
    assert "density" in result.columns


def test_validate_field_types_passes_for_canonical_df():
    df = _make_canonical_df()
    result = FieldMapper.validate_field_types(df)
    for field in ["formation_energy_per_atom", "energy_above_hull", "band_gap", "density"]:
        assert field in result
        assert result[field] is True


def test_validate_field_types_returns_false_for_missing_field():
    df = pd.DataFrame({"density": [7.8, 8.2]})
    result = FieldMapper.validate_field_types(df)
    assert result.get("formation_energy_per_atom") is False


def test_get_field_status_report_structure():
    df = _make_canonical_df()
    report = FieldMapper.get_field_status_report(df)
    assert "total_columns" in report
    assert "canonical_fields_found" in report
    assert "missing_canonical_fields" in report
    assert isinstance(report["canonical_fields_found"], list)
    assert "formation_energy_per_atom" in report["canonical_fields_found"]


def test_get_field_status_report_empty_df():
    df = pd.DataFrame()
    report = FieldMapper.get_field_status_report(df)
    assert report["total_columns"] == 0
    assert report["canonical_fields_found"] == []


def test_create_fallback_data_returns_correct_length_for_all_classifier_fields():
    for field in REQUIRED_FIELDS_ML_CLASSIFIER:
        data = FieldMapper.create_fallback_data(field, 10)
        assert len(data) == 10


def test_create_fallback_data_for_element_field():
    data = FieldMapper.create_fallback_data("element_1", 5)
    assert len(data) == 5


def test_create_fallback_data_for_formula_field():
    data = FieldMapper.create_fallback_data("formula", 4)
    assert len(data) == 4


def test_create_fallback_data_unknown_field_uses_default():
    data = FieldMapper.create_fallback_data("totally_unknown_field", 8)
    assert len(data) == 8


def test_create_fallback_data_formation_energy_clipped():
    data = FieldMapper.create_fallback_data("formation_energy_per_atom", 500)
    arr = np.asarray(data, dtype=float)
    assert arr.min() >= -6.0
    assert arr.max() <= 2.0


def test_create_fallback_data_band_gap_non_negative():
    data = FieldMapper.create_fallback_data("band_gap", 500)
    arr = np.asarray(data, dtype=float)
    assert arr.min() >= 0.0


def test_validate_dataframe_consistency_passes_with_required_fields():
    df = _make_canonical_df()
    is_valid = validate_dataframe_consistency(
        df, "test_context", required_fields=REQUIRED_FIELDS_ML_CLASSIFIER
    )
    assert is_valid is True


def test_validate_dataframe_consistency_fails_with_missing_fields():
    df = pd.DataFrame({"density": [7.8, 8.2]})
    is_valid = validate_dataframe_consistency(
        df, "test_context", required_fields=REQUIRED_FIELDS_ML_CLASSIFIER
    )
    assert is_valid is False


def test_validate_dataframe_consistency_full_canonical_check():
    df = _make_canonical_df()
    is_valid = validate_dataframe_consistency(df, "test_full")
    assert isinstance(is_valid, bool)


def test_standardize_dataframe_returns_unchanged_empty_df():
    df = pd.DataFrame()
    result = standardize_dataframe(df, REQUIRED_FIELDS_ML_CLASSIFIER, "test_empty")
    assert result.empty


def test_standardize_dataframe_with_alternate_names():
    df = pd.DataFrame(
        {
            "formation_energy": [-1.0, -2.0],
            "e_above_hull": [0.02, 0.08],
            "bandgap": [0.0, 1.0],
            "num_sites": [4, 6],
            "mass_density": [7.8, 8.2],
            "electronegativity_avg": [1.8, 1.9],
            "atomic_radius_avg": [1.3, 1.4],
        }
    )
    result = standardize_dataframe(df, REQUIRED_FIELDS_ML_CLASSIFIER, "test_alt")
    for field in REQUIRED_FIELDS_ML_CLASSIFIER:
        assert field in result.columns
