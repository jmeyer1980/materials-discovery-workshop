import pandas as pd

from materials_discovery_api import _extract_top_composition_fractions


def test_extract_top_composition_fractions_from_formula():
    row = pd.Series({"formula": "Al2Cu"})
    fractions = _extract_top_composition_fractions(row)

    assert len(fractions) == 3
    assert abs(sum(fractions) - 1.0) < 1e-6
    assert fractions[0] == 2 / 3
    assert fractions[1] == 1 / 3
    assert fractions[2] == 0.0


def test_extract_top_composition_fractions_fallback_to_elements():
    row = pd.Series({"elements": ["Al", "Ti", "Ni"]})
    fractions = _extract_top_composition_fractions(row)

    assert len(fractions) == 3
    assert all(abs(value - (1 / 3)) < 1e-6 for value in fractions)
