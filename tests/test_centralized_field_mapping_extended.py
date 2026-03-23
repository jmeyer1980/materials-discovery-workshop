import pandas as pd

from centralized_field_mapping import (
    REQUIRED_FIELDS,
    apply_field_mapping_to_generation,
    create_comprehensive_field_mapping_report,
    ensure_required_fields,
    find_field_name,
    generate_synthetic_field,
    safe_field_access,
    validate_dataframe_fields,
)


def _make_full_df(n=3):
    return pd.DataFrame(
        {
            "formation_energy_per_atom": [-1.0, -2.0, -0.5][:n],
            "energy_above_hull": [0.01, 0.05, 0.15][:n],
            "band_gap": [0.0, 1.2, 4.5][:n],
            "nsites": [4, 6, 8][:n],
            "density": [7.8, 8.2, 5.0][:n],
            "electronegativity": [1.8, 1.9, 1.7][:n],
            "atomic_radius": [1.3, 1.4, 1.2][:n],
        }
    )


def test_find_field_name_returns_none_when_no_match():
    result = find_field_name(["nonexistent_field", "also_not_here"], ["col_a", "col_b"])
    assert result is None


def test_generate_synthetic_field_yields_correct_shape_per_type():
    for field in [
        "formation_energy_per_atom",
        "energy_above_hull",
        "band_gap",
        "nsites",
        "density",
        "electronegativity",
        "atomic_radius",
    ]:
        arr = generate_synthetic_field(field, 10)
        assert len(arr) == 10


def test_generate_synthetic_field_fallback_for_unknown_field():
    arr = generate_synthetic_field("unknown_field_xyz", 5)
    assert len(arr) == 5


def test_ensure_required_fields_does_not_duplicate_when_already_present():
    df = _make_full_df()
    result = ensure_required_fields(df)
    for field in REQUIRED_FIELDS:
        assert field in result.columns
    assert len(result.columns) == len(set(result.columns))


def test_ensure_required_fields_fills_nan_with_defaults():
    df = pd.DataFrame(
        {
            "formation_energy_per_atom": [None, -1.0],
            "energy_above_hull": [None, 0.05],
            "band_gap": [None, 1.0],
            "nsites": [None, 4],
            "density": [None, 7.8],
            "electronegativity": [None, 1.8],
            "atomic_radius": [None, 1.3],
        }
    )
    result = ensure_required_fields(df)
    assert result["band_gap"].isna().sum() == 0
    assert result["density"].isna().sum() == 0


def test_validate_dataframe_fields_passes_for_valid_df():
    df = _make_full_df()
    result = validate_dataframe_fields(df)
    assert result["all_fields_present"] is True
    assert result["missing_fields"] == []
    assert result["total_rows"] == 3


def test_validate_dataframe_fields_reports_missing_fields():
    df = pd.DataFrame({"band_gap": [0.0, 1.0], "density": [7.8, 8.0]})
    result = validate_dataframe_fields(df)
    assert result["all_fields_present"] is False
    assert "formation_energy_per_atom" in result["missing_fields"]


def test_validate_dataframe_fields_column_count():
    df = _make_full_df()
    result = validate_dataframe_fields(df)
    assert result["total_columns"] == len(REQUIRED_FIELDS)
    assert result["total_rows"] == 3


def test_safe_field_access_existing_field():
    df = _make_full_df()
    series = safe_field_access(df, "band_gap")
    assert list(series) == list(df["band_gap"])


def test_safe_field_access_missing_field_with_default():
    df = pd.DataFrame({"density": [5.0, 6.0]})
    series = safe_field_access(df, "band_gap", default_value=0.0)
    assert list(series) == [0.0, 0.0]


def test_safe_field_access_missing_field_generates_synthetic():
    df = pd.DataFrame({"density": [5.0, 6.0, 7.0]})
    series = safe_field_access(df, "formation_energy_per_atom")
    assert len(series) == 3
    assert series.notna().all()


def test_apply_field_mapping_to_generation_with_alternate_names():
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
    result = apply_field_mapping_to_generation(df)
    for field in REQUIRED_FIELDS:
        assert field in result.columns


def test_create_comprehensive_field_mapping_report_contains_all_required_fields():
    df = _make_full_df()
    report = create_comprehensive_field_mapping_report(df)
    assert isinstance(report, str)
    assert "FIELD MAPPING REPORT" in report
    for field in REQUIRED_FIELDS:
        assert field in report


def test_create_comprehensive_field_mapping_report_flags_missing_field():
    df = pd.DataFrame({"density": [7.8, 8.2]})
    report = create_comprehensive_field_mapping_report(df)
    assert "MISSING" in report
