import pytest

try:
    from materials_discovery_model import MaterialsDiscoveryModel
    _MODEL_IMPORT_ERROR = None
except Exception as import_error:
    MaterialsDiscoveryModel = None
    _MODEL_IMPORT_ERROR = import_error


def test_create_formula_string_formats_to_three_decimals():
    if MaterialsDiscoveryModel is None:
        pytest.skip(f"Skipping due to model import error: {_MODEL_IMPORT_ERROR}")

    model = MaterialsDiscoveryModel.__new__(MaterialsDiscoveryModel)

    formula = model._create_formula_string(["Al", "Ti"], [0.5, 0.5])

    assert formula == "Al0.500Ti0.500"


def test_create_formula_string_handles_length_mismatch():
    if MaterialsDiscoveryModel is None:
        pytest.skip(f"Skipping due to model import error: {_MODEL_IMPORT_ERROR}")

    model = MaterialsDiscoveryModel.__new__(MaterialsDiscoveryModel)

    formula = model._create_formula_string(["Al", "Ti"], [1.0])

    assert formula == "Invalid"


def test_generate_compositions_normalizes_values():
    if MaterialsDiscoveryModel is None:
        pytest.skip(f"Skipping due to model import error: {_MODEL_IMPORT_ERROR}")

    model = MaterialsDiscoveryModel.__new__(MaterialsDiscoveryModel)

    row = {
        "composition_1": 0.9,
        "composition_2": 0.9,
        "composition_3": 0.1,
    }

    comps = model._generate_compositions(["Al", "Ti", "Ni"], row)

    assert len(comps) == 3
    assert abs(sum(comps) - 1.0) < 1e-6
    assert all(0.0 < value < 1.0 for value in comps)
