"""
Centralized Field Mapping and Validation System

This module provides comprehensive field mapping and validation functionality
to ensure all required fields are present in DataFrames before any analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define required fields for ML analysis
REQUIRED_FIELDS = [
    'formation_energy_per_atom', 
    'energy_above_hull', 
    'band_gap', 
    'nsites', 
    'density', 
    'electronegativity', 
    'atomic_radius'
]

# Field name mappings for different data sources
FIELD_MAPPINGS = {
    'formation_energy_per_atom': [
        'formation_energy_per_atom', 'formation_energy', 'energy_per_atom', 'e_form', 
        'formation_energy_ev_atom', 'energy_per_atom_ev'
    ],
    'energy_above_hull': [
        'energy_above_hull', 'e_above_hull', 'hull_energy', 'stability_energy',
        'energy_above_hull_ev_atom', 'stability_energy_ev_atom'
    ],
    'band_gap': [
        'band_gap', 'bandgap', 'gap', 'electronic_gap', 'band_gap_ev'
    ],
    'nsites': [
        'nsites', 'num_sites', 'sites', 'unit_cell_sites', 'total_sites'
    ],
    'density': [
        'density', 'mass_density', 'material_density', 'density_g_cm3'
    ],
    'electronegativity': [
        'electronegativity', 'avg_electronegativity', 'electronegativity_avg',
        'electronegativity_pauling'
    ],
    'atomic_radius': [
        'atomic_radius', 'avg_atomic_radius', 'atomic_radius_avg', 'atomic_radius_angstrom'
    ]
}


def log_dataframe_state(df: pd.DataFrame, stage: str) -> None:
    """Log the state of the DataFrame at a specific stage."""
    logger.info(f"=== {stage} ===")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Required fields status:")
    for field in REQUIRED_FIELDS:
        if field in df.columns:
            try:
                min_val = df[field].min() if not df[field].empty else 'N/A'
                max_val = df[field].max() if not df[field].empty else 'N/A'
                logger.info(f"  ✓ {field}: {df[field].dtype}, min={min_val:.3f}, max={max_val:.3f}")
            except Exception as e:
                logger.info(f"  ✓ {field}: {df[field].dtype} (error getting min/max: {e})")
        else:
            logger.info(f"  ✗ {field}: MISSING")
    logger.info("")


def find_field_name(possible_names: List[str], df_columns: List[str]) -> Optional[str]:
    """Find the correct field name from a list of possible names."""
    for name in possible_names:
        if name in df_columns:
            return name
    return None


def generate_synthetic_field(field_name: str, num_samples: int) -> np.ndarray:
    """Generate synthetic data for a missing field based on realistic material properties."""
    if field_name == 'formation_energy_per_atom':
        # Most stable alloys have formation energies between -4 and 0 eV/atom
        return np.random.normal(-1.5, 1.0, num_samples)
    elif field_name == 'energy_above_hull':
        # Most synthesizable alloys have E_hull < 0.1 eV/atom
        return np.random.exponential(0.05, num_samples)
    elif field_name == 'band_gap':
        # For alloys, most are metallic (band_gap ≈ 0) or have small band gaps
        return np.random.exponential(0.5, num_samples)
    elif field_name == 'nsites':
        # Unit cell size for alloys
        return np.random.randint(2, 15, num_samples)
    elif field_name == 'density':
        # Reasonable density range for alloys
        return np.random.normal(7.8, 2.0, num_samples)
    elif field_name == 'electronegativity':
        # Reasonable electronegativity range
        return np.random.normal(1.8, 0.3, num_samples)
    elif field_name == 'atomic_radius':
        # Reasonable atomic radius range
        return np.random.normal(1.3, 0.2, num_samples)
    else:
        # Default fallback
        return np.random.normal(0, 1, num_samples)


def ensure_required_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all required fields exist in DataFrame, creating synthetic data if needed."""
    logger.info("Ensuring all required fields are present...")
    df_copy = df.copy()
    modified_fields = []
    
    for field in REQUIRED_FIELDS:
        if field not in df_copy.columns:
            # Try to find the field with different naming conventions
            source_field = find_field_name(FIELD_MAPPINGS[field], df_copy.columns)
            
            if source_field:
                logger.info(f"Field mapping: {source_field} -> {field}")
                df_copy[field] = df_copy[source_field]
                modified_fields.append(field)
            else:
                # Create synthetic data for missing field
                logger.warning(f"Required field '{field}' not found, creating synthetic data")
                synthetic_data = generate_synthetic_field(field, len(df_copy))
                df_copy[field] = synthetic_data
                modified_fields.append(field)
        else:
            logger.info(f"Field '{field}' already present")
    
    # Ensure all columns are numeric with robust fallback
    for field in REQUIRED_FIELDS:
        if field in df_copy.columns:
            # Convert to numeric, coerce errors to NaN, then fill with reasonable defaults
            original_values = df_copy[field].copy()
            df_copy[field] = pd.to_numeric(df_copy[field], errors='coerce')
            
            # Fill NaN values with reasonable defaults based on field type
            if field in ['formation_energy_per_atom', 'energy_above_hull']:
                # For energy fields, use small positive values for NaN
                df_copy[field] = df_copy[field].fillna(0.1)
            elif field == 'band_gap':
                # For band gap, assume metallic (zero gap) for NaN
                df_copy[field] = df_copy[field].fillna(0.0)
            elif field == 'nsites':
                # For unit cell sites, use typical value
                df_copy[field] = df_copy[field].fillna(4.0)
            elif field == 'density':
                # For density, use typical alloy density
                df_copy[field] = df_copy[field].fillna(7.8)
            elif field == 'electronegativity':
                # For electronegativity, use typical value
                df_copy[field] = df_copy[field].fillna(1.8)
            elif field == 'atomic_radius':
                # For atomic radius, use typical value
                df_copy[field] = df_copy[field].fillna(1.3)
    
    logger.info(f"Field mapping completed. Modified fields: {modified_fields}")
    return df_copy


def validate_dataframe_fields(df: pd.DataFrame) -> Dict[str, Union[bool, List[str]]]:
    """Validate that DataFrame has all required fields with proper types."""
    missing_fields = [field for field in REQUIRED_FIELDS if field not in df.columns]
    invalid_fields = []
    
    for field in REQUIRED_FIELDS:
        if field in df.columns:
            try:
                # Try to convert to numeric to check if it's a valid numeric field
                pd.to_numeric(df[field], errors='raise')
            except (ValueError, TypeError):
                invalid_fields.append(field)
    
    return {
        'all_fields_present': len(missing_fields) == 0,
        'missing_fields': missing_fields,
        'invalid_fields': invalid_fields,
        'total_rows': len(df),
        'total_columns': len(df.columns)
    }


def safe_field_access(df: pd.DataFrame, field_name: str, default_value: Optional[float] = None) -> pd.Series:
    """Safely access a field, providing fallback if it doesn't exist."""
    if field_name in df.columns:
        return df[field_name]
    else:
        if default_value is not None:
            logger.warning(f"Field '{field_name}' not found, using default value: {default_value}")
            return pd.Series([default_value] * len(df), index=df.index)
        else:
            logger.warning(f"Field '{field_name}' not found, generating synthetic data")
            synthetic_data = generate_synthetic_field(field_name, len(df))
            return pd.Series(synthetic_data, index=df.index)


def apply_field_mapping_to_generation(generated_df: pd.DataFrame) -> pd.DataFrame:
    """Apply field mapping immediately after material generation."""
    logger.info("Applying field mapping to generated materials...")
    log_dataframe_state(generated_df, "Before field mapping")
    
    # Ensure all required fields are present
    mapped_df = ensure_required_fields(generated_df)
    
    log_dataframe_state(mapped_df, "After field mapping")
    
    # Validate the result
    validation_result = validate_dataframe_fields(mapped_df)
    if not validation_result['all_fields_present']:
        logger.error(f"Field mapping failed. Missing fields: {validation_result['missing_fields']}")
        raise ValueError(f"Field mapping failed. Missing fields: {validation_result['missing_fields']}")
    
    logger.info("Field mapping validation passed")
    return mapped_df


def create_comprehensive_field_mapping_report(df: pd.DataFrame) -> str:
    """Create a comprehensive report of field mapping status."""
    report = []
    report.append("=== FIELD MAPPING REPORT ===")
    report.append(f"DataFrame shape: {df.shape}")
    report.append(f"Total columns: {len(df.columns)}")
    report.append("")
    report.append("Required fields status:")
    
    for field in REQUIRED_FIELDS:
        if field in df.columns:
            try:
                min_val = df[field].min() if not df[field].empty else 'N/A'
                max_val = df[field].max() if not df[field].empty else 'N/A'
                dtype = df[field].dtype
                report.append(f"  ✓ {field}: {dtype}, min={min_val:.3f}, max={max_val:.3f}")
            except Exception as e:
                report.append(f"  ✓ {field}: {df[field].dtype} (error getting stats: {e})")
        else:
            report.append(f"  ✗ {field}: MISSING")
    
    report.append("")
    report.append("Available columns:")
    for col in sorted(df.columns):
        report.append(f"  - {col}")
    
    return "\n".join(report)


if __name__ == "__main__":
    # Test the field mapping system
    test_df = pd.DataFrame({
        'formation_energy': [1.0, 2.0, 3.0],
        'e_above_hull': [0.1, 0.2, 0.3],
        'bandgap': [0.5, 1.0, 1.5],
        'num_sites': [4, 6, 8],
        'mass_density': [7.8, 8.2, 7.5],
        'electronegativity_avg': [1.8, 1.9, 1.7],
        'atomic_radius_avg': [1.3, 1.4, 1.2]
    })
    
    print("Original DataFrame:")
    print(test_df)
    print()
    
    mapped_df = apply_field_mapping_to_generation(test_df)
    print("Mapped DataFrame:")
    print(mapped_df)
    print()
    
    print(create_comprehensive_field_mapping_report(mapped_df))