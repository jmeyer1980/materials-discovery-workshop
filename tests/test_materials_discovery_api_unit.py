import os

import numpy as np
import pandas as pd
import pytest

from materials_discovery_api import (
    MaterialsProjectClient,
    _extract_top_composition_fractions,
    validate_data_with_schema,
)

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), '..', 'feature_schema.yml')


# ---------------------------------------------------------------------------
# _extract_top_composition_fractions
# ---------------------------------------------------------------------------


def test_extract_top_composition_fractions_from_formula():
    row = pd.Series({'formula': 'Al2Cu'})
    fractions = _extract_top_composition_fractions(row)

    assert len(fractions) == 3
    assert abs(sum(fractions) - 1.0) < 1e-6
    assert fractions[0] == 2 / 3
    assert fractions[1] == 1 / 3
    assert fractions[2] == 0.0


def test_extract_top_composition_fractions_fallback_to_elements():
    row = pd.Series({'elements': ['Al', 'Ti', 'Ni']})
    fractions = _extract_top_composition_fractions(row)

    assert len(fractions) == 3
    assert all(abs(value - (1 / 3)) < 1e-6 for value in fractions)


# ---------------------------------------------------------------------------
# validate_data_with_schema
# ---------------------------------------------------------------------------


def test_validate_data_with_schema_missing_file_returns_unchanged():
    df = pd.DataFrame({'composition_1': [0.5, 1.5], 'composition_2': [0.5, -0.1]})
    result = validate_data_with_schema(df, '/nonexistent/path/schema.yml')
    pd.testing.assert_frame_equal(result, df)


def test_validate_data_with_schema_clips_out_of_range_values():
    df = pd.DataFrame({'composition_1': [-0.5, 0.5, 2.0], 'composition_2': [0.3, 0.5, 0.7]})
    result = validate_data_with_schema(df, _SCHEMA_PATH)
    assert result['composition_1'].min() >= 0.0
    assert result['composition_1'].max() <= 1.0


def test_validate_data_with_schema_fills_nan_with_mean():
    df = pd.DataFrame({'composition_1': [0.2, np.nan, 0.6]})
    result = validate_data_with_schema(df, _SCHEMA_PATH)
    assert result['composition_1'].isna().sum() == 0
    assert abs(result['composition_1'].iloc[1] - 0.4) < 1e-6


def test_validate_data_with_schema_unknown_column_skipped():
    df = pd.DataFrame({'unknown_col': [99.9, -5.0]})
    result = validate_data_with_schema(df, _SCHEMA_PATH)
    assert list(result['unknown_col']) == [99.9, -5.0]


# ---------------------------------------------------------------------------
# MaterialsProjectClient
# ---------------------------------------------------------------------------


def test_materials_project_client_raises_without_key():
    with pytest.raises(ValueError, match='API key'):
        MaterialsProjectClient(api_key=None)


def test_materials_project_client_raises_with_empty_string_key():
    with pytest.raises(ValueError, match='API key'):
        MaterialsProjectClient(api_key='')
