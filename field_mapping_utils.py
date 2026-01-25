"""
Field Mapping Utilities for Materials Discovery Workshop
=========================================================

This module provides utilities to ensure consistent field naming across all modules
in the materials discovery pipeline. It handles field name variations that might
occur in different environments (Docker, cloud, local) and provides robust
field validation and mapping.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class FieldMapper:
    """Utility class for consistent field mapping across modules."""

    # Canonical field names as defined in feature_schema.yml
    CANONICAL_FIELDS = {
        'formation_energy_per_atom': 'formation_energy_per_atom',
        'energy_above_hull': 'energy_above_hull',
        'band_gap': 'band_gap',
        'nsites': 'nsites',
        'density': 'density',
        'electronegativity': 'electronegativity',
        'atomic_radius': 'atomic_radius',
        'composition_1': 'composition_1',
        'composition_2': 'composition_2',
        'composition_3': 'composition_3',
        'element_1': 'element_1',
        'element_2': 'element_2',
        'element_3': 'element_3',
        'formula': 'formula',
        'melting_point': 'melting_point'
    }

    # Common field name variations that might occur
    FIELD_VARIATIONS = {
        'formation_energy_per_atom': [
            'formation_energy_per_atom', 'formation_energy', 'energy_per_atom',
            'e_form', 'formation_energy_atom', 'energy_formation_per_atom'
        ],
        'energy_above_hull': [
            'energy_above_hull', 'e_above_hull', 'hull_energy', 'stability_energy',
            'energy_hull', 'above_hull', 'energy_above_convex_hull'
        ],
        'band_gap': [
            'band_gap', 'bandgap', 'gap', 'electronic_gap', 'band_gap_eV',
            'electronic_band_gap', 'optical_gap'
        ],
        'nsites': [
            'nsites', 'num_sites', 'sites', 'unit_cell_sites', 'n_sites',
            'number_of_sites', 'total_sites'
        ],
        'density': [
            'density', 'mass_density', 'material_density', 'bulk_density',
            'density_g_cm3', 'density_g_per_cm3'
        ],
        'electronegativity': [
            'electronegativity', 'avg_electronegativity', 'electronegativity_avg',
            'electronegativity_average', 'pauling_electronegativity'
        ],
        'atomic_radius': [
            'atomic_radius', 'avg_atomic_radius', 'atomic_radius_avg',
            'atomic_radius_average', 'covalent_radius', 'atomic_size'
        ],
        'composition_1': [
            'composition_1', 'comp_1', 'composition1', 'atomic_fraction_1',
            'element1_fraction', 'fraction_1'
        ],
        'composition_2': [
            'composition_2', 'comp_2', 'composition2', 'atomic_fraction_2',
            'element2_fraction', 'fraction_2'
        ],
        'composition_3': [
            'composition_3', 'comp_3', 'composition3', 'atomic_fraction_3',
            'element3_fraction', 'fraction_3'
        ],
        'element_1': [
            'element_1', 'elem_1', 'element1', 'first_element', 'primary_element'
        ],
        'element_2': [
            'element_2', 'elem_2', 'element2', 'second_element', 'secondary_element'
        ],
        'element_3': [
            'element_3', 'elem_3', 'element3', 'third_element', 'tertiary_element'
        ],
        'formula': [
            'formula', 'chemical_formula', 'formula_pretty', 'composition_formula'
        ],
        'melting_point': [
            'melting_point', 'melting_temp', 'mp', 'melting_temperature'
        ]
    }

    @classmethod
    def find_canonical_field(cls, possible_names: List[str], df_columns: List[str]) -> Optional[str]:
        """
        Find the canonical field name from a list of possible names.

        Args:
            possible_names: List of possible field names
            df_columns: List of columns in the DataFrame

        Returns:
            Canonical field name if found, None otherwise
        """
        for name in possible_names:
            if name in df_columns:
                return name
        return None

    @classmethod
    def map_fields_to_canonical(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map all fields in a DataFrame to canonical names.

        Args:
            df: DataFrame with potentially non-canonical field names

        Returns:
            DataFrame with canonical field names
        """
        df_mapped = df.copy()
        mapping_applied = {}

        for canonical_name, variations in cls.FIELD_VARIATIONS.items():
            found_field = cls.find_canonical_field(variations, df.columns.tolist())
            if found_field and found_field != canonical_name:
                df_mapped = df_mapped.rename(columns={found_field: canonical_name})
                mapping_applied[found_field] = canonical_name

        if mapping_applied:
            print(f"Field mapping applied: {mapping_applied}")

        return df_mapped

    @classmethod
    def ensure_required_fields(cls, df: pd.DataFrame, required_fields: List[str]) -> pd.DataFrame:
        """
        Ensure all required fields are present in the DataFrame.

        Args:
            df: DataFrame to check
            required_fields: List of required field names

        Returns:
            DataFrame with all required fields
        """
        df_result = df.copy()
        missing_fields = []

        for field in required_fields:
            if field not in df_result.columns:
                missing_fields.append(field)

        if missing_fields:
            print(f"Warning: Missing required fields: {missing_fields}")
            print(f"Available fields: {list(df_result.columns)}")

            # Try to find and map missing fields
            for field in missing_fields:
                if field in cls.FIELD_VARIATIONS:
                    found_field = cls.find_canonical_field(
                        cls.FIELD_VARIATIONS[field],
                        df_result.columns.tolist()
                    )
                    if found_field:
                        df_result = df_result.rename(columns={found_field: field})
                        print(f"  Mapped '{found_field}' -> '{field}'")
                    else:
                        print(f"  Could not find field '{field}' or its variations")
                else:
                    print(f"  No variations defined for field '{field}'")

        return df_result

    @classmethod
    def validate_field_types(cls, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that fields have appropriate data types.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with field names as keys and validation results as values
        """
        validation_results = {}

        # Define expected types for key fields
        expected_types = {
            'formation_energy_per_atom': [float, int, np.float64, np.float32],
            'energy_above_hull': [float, int, np.float64, np.float32],
            'band_gap': [float, int, np.float64, np.float32],
            'nsites': [int, np.int64, np.int32],
            'density': [float, int, np.float64, np.float32],
            'electronegativity': [float, int, np.float64, np.float32],
            'atomic_radius': [float, int, np.float64, np.float32],
            'composition_1': [float, int, np.float64, np.float32],
            'composition_2': [float, int, np.float64, np.float32],
            'composition_3': [float, int, np.float64, np.float32]
        }

        for field, expected_types_list in expected_types.items():
            if field in df.columns:
                actual_type = type(df[field].iloc[0]) if len(df) > 0 else None
                is_valid = any(isinstance(df[field].iloc[0], expected_type)
                             for expected_type in expected_types_list) if len(df) > 0 else False
                validation_results[field] = is_valid

                if not is_valid and len(df) > 0:
                    print(f"Warning: Field '{field}' has unexpected type {actual_type}")
            else:
                validation_results[field] = False

        return validation_results

    @classmethod
    def get_field_status_report(cls, df: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive field status report.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with field status information
        """
        report = {
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'canonical_fields_found': [],
            'missing_canonical_fields': [],
            'field_types': {},
            'validation_results': {}
        }

        # Check for canonical fields
        for canonical_field in cls.CANONICAL_FIELDS.values():
            if canonical_field in df.columns:
                report['canonical_fields_found'].append(canonical_field)
            else:
                report['missing_canonical_fields'].append(canonical_field)

        # Get field types
        for col in df.columns:
            if len(df) > 0:
                report['field_types'][col] = str(type(df[col].iloc[0]).__name__)

        # Run validation
        report['validation_results'] = cls.validate_field_types(df)

        return report

    @classmethod
    def create_fallback_data(cls, field_name: str, size: int) -> np.ndarray:
        """
        Create fallback data for missing fields based on field type.

        Args:
            field_name: Name of the field
            size: Number of samples needed

        Returns:
            Array with fallback data
        """
        # Define fallback strategies for different field types
        fallback_strategies = {
            'formation_energy_per_atom': lambda n: np.random.normal(-1.5, 1.0, n),
            'energy_above_hull': lambda n: np.random.exponential(0.05, n),
            'band_gap': lambda n: np.random.exponential(0.5, n),
            'nsites': lambda n: np.random.randint(2, 15, n),
            'density': lambda n: np.random.normal(7.8, 2.0, n),
            'electronegativity': lambda n: np.random.normal(1.8, 0.3, n),
            'atomic_radius': lambda n: np.random.normal(1.3, 0.2, n),
            'composition_1': lambda n: np.random.uniform(0.05, 0.95, n),
            'composition_2': lambda n: np.random.uniform(0.05, 0.95, n),
            'composition_3': lambda n: np.zeros(n),
            'element_1': lambda n: ['Al'] * n,
            'element_2': lambda n: ['Ti'] * n,
            'element_3': lambda n: [None] * n,
            'formula': lambda n: [f'Al{np.random.uniform(0.1, 0.9):.3f}Ti{np.random.uniform(0.1, 0.9):.3f}'] * n,
            'melting_point': lambda n: np.random.normal(1500, 300, n)
        }

        if field_name in fallback_strategies:
            data = fallback_strategies[field_name](size)
            # Apply constraints for specific fields
            if field_name == 'formation_energy_per_atom':
                data = np.clip(data, -6.0, 2.0)
            elif field_name == 'energy_above_hull':
                data = np.clip(data, 0, 0.5)
            elif field_name == 'band_gap':
                data = np.clip(data, 0, 8.0)
            elif field_name == 'density':
                data = np.clip(data, 2.0, 25.0)
            elif field_name == 'electronegativity':
                data = np.clip(data, 0.7, 2.5)
            elif field_name == 'atomic_radius':
                data = np.clip(data, 0.8, 2.2)
            elif field_name in ['composition_1', 'composition_2']:
                data = np.clip(data, 0.05, 0.95)
            elif field_name == 'melting_point':
                data = np.maximum(data, 500)

            return data
        else:
            # Default fallback
            return np.random.normal(0, 1, size)

    @classmethod
    def ensure_field_completeness(cls, df: pd.DataFrame, required_fields: List[str]) -> pd.DataFrame:
        """
        Ensure DataFrame has all required fields with valid data.

        Args:
            df: DataFrame to process
            required_fields: List of required field names

        Returns:
            DataFrame with all required fields
        """
        df_result = df.copy()

        # First, try to map existing fields to canonical names
        df_result = cls.map_fields_to_canonical(df_result)

        # Check for missing fields and create fallbacks
        for field in required_fields:
            if field not in df_result.columns:
                print(f"Creating fallback data for missing field: {field}")
                fallback_data = cls.create_fallback_data(field, len(df_result))
                df_result[field] = fallback_data

        # Ensure data types are correct
        for field in required_fields:
            if field in df_result.columns:
                if field in ['nsites']:
                    # Integer fields
                    df_result[field] = pd.to_numeric(df_result[field], errors='coerce').fillna(0).astype(int)
                else:
                    # Float fields
                    df_result[field] = pd.to_numeric(df_result[field], errors='coerce').fillna(0.0)

        return df_result


def validate_dataframe_consistency(df: pd.DataFrame, context: str = "unknown") -> bool:
    """
    Validate that a DataFrame has consistent field naming and data types.

    Args:
        df: DataFrame to validate
        context: Context string for logging (e.g., "gradio_app", "synthesizability_predictor")

    Returns:
        True if validation passes, False otherwise
    """
    print(f"\n=== Validating DataFrame consistency for {context} ===")

    # Get field status report
    report = FieldMapper.get_field_status_report(df)

    print(f"Total columns: {report['total_columns']}")
    print(f"Canonical fields found: {len(report['canonical_fields_found'])}")
    print(f"Missing canonical fields: {len(report['missing_canonical_fields'])}")

    if report['missing_canonical_fields']:
        print(f"Missing fields: {report['missing_canonical_fields']}")

    # Check validation results
    invalid_fields = [field for field, is_valid in report['validation_results'].items() if not is_valid]
    if invalid_fields:
        print(f"Invalid field types: {invalid_fields}")

    # Overall validation result
    has_required_fields = len(report['missing_canonical_fields']) == 0
    all_types_valid = len(invalid_fields) == 0

    validation_passed = has_required_fields and all_types_valid

    print(f"Validation result: {'PASS' if validation_passed else 'FAIL'}")
    print("=" * 50)

    return validation_passed


def standardize_dataframe(df: pd.DataFrame, required_fields: List[str], context: str = "unknown") -> pd.DataFrame:
    """
    Standardize a DataFrame to ensure consistent field naming and data types.

    Args:
        df: DataFrame to standardize
        required_fields: List of required field names
        context: Context string for logging

    Returns:
        Standardized DataFrame
    """
    print(f"\n=== Standardizing DataFrame for {context} ===")

    if df.empty:
        print("Warning: Empty DataFrame provided")
        return df

    # Step 1: Map to canonical field names
    df_standardized = FieldMapper.map_fields_to_canonical(df)

    # Step 2: Ensure all required fields are present
    df_standardized = FieldMapper.ensure_field_completeness(df_standardized, required_fields)

    # Step 3: Validate the result
    is_valid = validate_dataframe_consistency(df_standardized, context)

    if not is_valid:
        print("Warning: Standardization completed but validation failed")
    else:
        print("Standardization completed successfully")

    return df_standardized


# Global field mapping instance for easy access
field_mapper = FieldMapper()

# Common required field sets for different modules
REQUIRED_FIELDS_GRADIO = [
    'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'nsites',
    'density', 'electronegativity', 'atomic_radius', 'composition_1', 'composition_2',
    'element_1', 'element_2', 'formula'
]

REQUIRED_FIELDS_ML_CLASSIFIER = [
    'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'nsites',
    'density', 'electronegativity', 'atomic_radius'
]

REQUIRED_FIELDS_VAE = [
    'composition_1', 'composition_2', 'formation_energy_per_atom', 'density',
    'electronegativity', 'atomic_radius'
]


if __name__ == "__main__":
    # Test the field mapping utilities
    print("Testing Field Mapping Utilities")
    print("=" * 40)

    # Create test DataFrame with non-canonical field names
    test_data = {
        'formation_energy': [-2.5, -1.0, 0.5],
        'e_above_hull': [0.02, 0.05, 0.8],
        'bandgap': [1.2, 0.0, 5.0],
        'num_sites': [8, 4, 32],
        'mass_density': [7.2, 4.5, 12.5],
        'electronegativity_avg': [1.8, 1.5, 2.2],
        'atomic_radius_avg': [1.4, 1.2, 1.8],
        'comp_1': [0.5, 0.6, 0.4],
        'comp_2': [0.5, 0.4, 0.6],
        'elem_1': ['Al', 'Ti', 'Fe'],
        'elem_2': ['Ti', 'Al', 'Ni'],
        'formula_pretty': ['Al0.5Ti0.5', 'Ti0.6Al0.4', 'Fe0.4Ni0.6']
    }

    test_df = pd.DataFrame(test_data)
    print("Original DataFrame columns:", list(test_df.columns))

    # Standardize the DataFrame
    standardized_df = standardize_dataframe(
        test_df,
        REQUIRED_FIELDS_GRADIO,
        "test_module"
    )

    print("Standardized DataFrame columns:", list(standardized_df.columns))
    print("Sample data:")
    print(standardized_df.head(1))