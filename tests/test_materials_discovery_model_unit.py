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


# ─────────────────────────────────────────────────────────────────────────────
# Shared import guard for classes that require torch
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import numpy as np
    import pandas as pd
    from materials_discovery_model import (
        FeatureEngineer,
        MaterialsDataset,
        VariationalAutoencoder,
    )
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


def _make_materials_df(n=4):
    """Minimal DataFrame with all columns FeatureEngineer expects."""
    return pd.DataFrame({
        "element_1": ["Al", "Al", "Ti", "Ti"],
        "element_2": ["Ti", "Cu", "Fe", "Ni"],
        "element_3": [None, None, None, None],
        "composition_1": [0.5, 0.4, 0.6, 0.3],
        "composition_2": [0.5, 0.6, 0.4, 0.7],
        "composition_3": [0.0, 0.0, 0.0, 0.0],
        "melting_point": [1500, 1200, 1800, 1450],
        "boiling_point": [3000, 2500, 3500, 3000],
        "density": [4.5, 5.0, 6.0, 7.0],
        "electronegativity": [1.6, 1.7, 1.5, 1.8],
        "atomic_radius": [1.4, 1.3, 1.45, 1.25],
        "electronegativity_difference": [0.1, 0.2, 0.15, 0.3],
    }).head(n)


# ─────────────────────────────────────────────────────────────────────────────
# FeatureEngineer
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_engineer_create_features_returns_dataframe_and_columns():
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    fe = FeatureEngineer()
    df, cols = fe.create_features(_make_materials_df())
    assert isinstance(cols, list)
    assert len(cols) > 0
    for col in cols:
        assert col in df.columns


def test_feature_engineer_create_features_adds_interaction_columns():
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    fe = FeatureEngineer()
    df, _ = fe.create_features(_make_materials_df())
    assert "comp_1_2" in df.columns
    assert "mp_bp_ratio" in df.columns
    assert "electroneg_sq" in df.columns


def test_feature_engineer_scale_features_fit_normalises():
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    fe = FeatureEngineer()
    df, cols = fe.create_features(_make_materials_df())
    scaled = fe.scale_features(df, cols, fit=True)
    # After StandardScaler fit+transform the mean of each feature should be ~0
    assert abs(scaled[cols].mean().mean()) < 1e-6


def test_feature_engineer_scale_features_transform_without_fit_raises():
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    fe = FeatureEngineer()
    df, cols = fe.create_features(_make_materials_df())
    with pytest.raises(ValueError, match="not fitted"):
        fe.scale_features(df, cols, fit=False)


# ─────────────────────────────────────────────────────────────────────────────
# MaterialsDataset
# ─────────────────────────────────────────────────────────────────────────────

def test_materials_dataset_len():
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    fe = FeatureEngineer()
    df, cols = fe.create_features(_make_materials_df(n=4))
    dataset = MaterialsDataset(df, cols)
    assert len(dataset) == 4


def test_materials_dataset_getitem_returns_features_tuple():
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    fe = FeatureEngineer()
    df, cols = fe.create_features(_make_materials_df(n=4))
    dataset = MaterialsDataset(df, cols)
    item = dataset[0]
    # No targets → returns (features,)
    assert isinstance(item, tuple)
    assert len(item) == 1
    assert item[0].shape[0] == len(cols)


def test_materials_dataset_getitem_with_targets():
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    fe = FeatureEngineer()
    df, cols = fe.create_features(_make_materials_df(n=4))
    df["target"] = 1.0
    dataset = MaterialsDataset(df, cols, target_columns=["target"])
    features, targets = dataset[0]
    assert features.shape[0] == len(cols)
    assert targets.shape[0] == 1


# ─────────────────────────────────────────────────────────────────────────────
# VariationalAutoencoder
# ─────────────────────────────────────────────────────────────────────────────

def test_vae_init_sets_dimensions():
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    vae = VariationalAutoencoder(input_dim=20, latent_dim=8, hidden_dims=[32, 16])
    assert vae.input_dim == 20
    assert vae.latent_dim == 8


def test_vae_forward_output_shape():
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    vae = VariationalAutoencoder(input_dim=10, latent_dim=4, hidden_dims=[16, 8])
    x = torch.randn(3, 10)
    reconstructed, mu, log_var = vae(x)
    assert reconstructed.shape == (3, 10)
    assert mu.shape == (3, 4)
    assert log_var.shape == (3, 4)


def test_vae_sample_returns_correct_shape():
    if not _TORCH_AVAILABLE:
        pytest.skip("torch not available")
    vae = VariationalAutoencoder(input_dim=10, latent_dim=4, hidden_dims=[16, 8])
    samples = vae.sample(num_samples=5)
    assert samples.shape == (5, 10)
