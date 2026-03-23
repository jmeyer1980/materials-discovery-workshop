import numpy as np

from data_generator import MaterialsDatasetGenerator


def test_init_loads_element_properties():
    gen = MaterialsDatasetGenerator(seed=0)
    assert len(gen.element_properties) > 0
    assert "Al" in gen.element_properties
    assert "electronegativity" in gen.element_properties["Al"]
    assert "melting_point" in gen.element_properties["Al"]


def test_init_elements_list_is_non_empty():
    gen = MaterialsDatasetGenerator(seed=0)
    assert len(gen.elements) > 10


def test_generate_binary_alloys_shape_and_columns():
    gen = MaterialsDatasetGenerator(seed=42)
    df = gen.generate_binary_alloys(n_samples=10)
    assert len(df) == 10
    for col in ["element_1", "element_2", "composition_1", "composition_2", "formula"]:
        assert col in df.columns


def test_generate_binary_alloys_compositions_sum_to_one():
    gen = MaterialsDatasetGenerator(seed=7)
    df = gen.generate_binary_alloys(n_samples=20)
    total = df["composition_1"] + df["composition_2"]
    assert np.allclose(total.to_numpy(dtype=float), 1.0, atol=1e-9)


def test_generate_binary_alloys_compositions_in_valid_range():
    gen = MaterialsDatasetGenerator(seed=1)
    df = gen.generate_binary_alloys(n_samples=50)
    assert (df["composition_1"] > 0).all()
    assert (df["composition_2"] > 0).all()
    assert (df["composition_1"] < 1).all()


def test_generate_binary_alloys_has_property_columns():
    gen = MaterialsDatasetGenerator(seed=99)
    df = gen.generate_binary_alloys(n_samples=5)
    for col in ["melting_point", "density", "electronegativity", "atomic_radius"]:
        assert col in df.columns


def test_generate_ternary_alloys_shape_and_columns():
    gen = MaterialsDatasetGenerator(seed=42)
    df = gen.generate_ternary_alloys(n_samples=10)
    assert len(df) == 10
    for col in ["element_3", "composition_3"]:
        assert col in df.columns


def test_generate_ternary_alloys_compositions_are_positive():
    gen = MaterialsDatasetGenerator(seed=3)
    df = gen.generate_ternary_alloys(n_samples=30)
    assert (df["composition_3"] > 0).all()


def test_generate_ternary_alloys_compositions_sum_near_one():
    gen = MaterialsDatasetGenerator(seed=5)
    df = gen.generate_ternary_alloys(n_samples=20)
    total = df["composition_1"] + df["composition_2"] + df["composition_3"]
    # Ternary compositions may be renormalized; all should be close to 1
    assert np.allclose(total.to_numpy(dtype=float), 1.0, atol=0.05)


def test_calculate_alloy_properties_binary_has_all_keys():
    gen = MaterialsDatasetGenerator(seed=0)
    props = gen._calculate_alloy_properties([("Al", 0.5), ("Ti", 0.5)])
    for key in ["melting_point", "density", "electronegativity", "atomic_radius", "electronegativity_difference"]:
        assert key in props


def test_calculate_alloy_properties_binary_delta_chi_non_negative():
    gen = MaterialsDatasetGenerator(seed=0)
    props = gen._calculate_alloy_properties([("Al", 0.5), ("Ti", 0.5)])
    assert props["electronegativity_difference"] >= 0


def test_calculate_alloy_properties_ternary_sets_delta_chi_to_zero():
    gen = MaterialsDatasetGenerator(seed=0)
    props = gen._calculate_alloy_properties([("Al", 0.4), ("Ti", 0.3), ("Fe", 0.3)])
    assert props["electronegativity_difference"] == 0


def test_generate_dataset_combines_binary_and_ternary():
    gen = MaterialsDatasetGenerator(seed=42)
    df = gen.generate_dataset(n_binary=5, n_ternary=3)
    assert len(df) == 8
    assert "alloy_type" in df.columns
    assert set(df["alloy_type"].unique()).issubset({"binary", "ternary"})


def test_generate_dataset_binary_and_ternary_counts():
    gen = MaterialsDatasetGenerator(seed=10)
    df = gen.generate_dataset(n_binary=8, n_ternary=4)
    assert len(df[df["alloy_type"] == "binary"]) == 8
    assert len(df[df["alloy_type"] == "ternary"]) == 4


def test_generate_binary_alloys_formula_contains_elements():
    gen = MaterialsDatasetGenerator(seed=42)
    df = gen.generate_binary_alloys(n_samples=5)
    for _, row in df.iterrows():
        assert row["element_1"] in row["formula"] or row["element_2"] in row["formula"]
