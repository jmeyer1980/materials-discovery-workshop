import numpy as np

from synthesizability_predictor import create_synthetic_dataset_fallback


def test_converted_initialization_synthetic_dataset_has_required_columns():
    dataset = create_synthetic_dataset_fallback(64)

    required_columns = {
        "composition_1",
        "composition_2",
        "composition_3",
        "formation_energy_per_atom",
        "energy_above_hull",
        "band_gap",
        "nsites",
        "density",
        "electronegativity",
        "atomic_radius",
    }

    assert len(dataset) == 64
    assert required_columns.issubset(set(dataset.columns))


def test_converted_initialization_feature_matrix_is_finite_numeric():
    dataset = create_synthetic_dataset_fallback(50)

    feature_cols = [
        "composition_1",
        "composition_2",
        "formation_energy_per_atom",
        "density",
        "electronegativity",
        "atomic_radius",
    ]

    matrix = dataset[feature_cols].to_numpy(dtype=float)

    assert matrix.shape == (50, len(feature_cols))
    assert np.isfinite(matrix).all()


def test_converted_initialization_compositions_sum_to_one():
    dataset = create_synthetic_dataset_fallback(80)

    composition_sum = (
        dataset["composition_1"] + dataset["composition_2"] + dataset["composition_3"]
    )

    assert np.allclose(composition_sum.to_numpy(dtype=float), 1.0, atol=1e-6)
