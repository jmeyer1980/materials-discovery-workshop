"""
Materials Project API Integration for Materials Discovery Workshop
====================================================================

This module provides functions to query real materials data from the Materials Project
database using their REST API. It replaces synthetic data generation with actual
materials science data for more realistic ML training and validation.

IMPORTANT: A valid Materials Project API key is required to use this module.
You can obtain a free API key from: https://materialsproject.org/api
Set the API key as an environment variable: export MP_API_KEY="your_key_here"
Or pass it directly when creating a MaterialsProjectClient instance.
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from typing import List, Dict, Tuple
import pymatgen.core as mg
import yaml

# Import field mapping utilities
from field_mapping_utils import (
    validate_dataframe_consistency,
    standardize_dataframe,
    REQUIRED_FIELDS_ML_CLASSIFIER
)


def validate_data_with_schema(df: pd.DataFrame, schema_path: str) -> pd.DataFrame:
    """
    Validates and cleans a DataFrame using a predefined YAML schema.

    - Checks for the presence of required columns.
    - Clips numeric columns to the valid range specified in the schema.
    - Fills missing values with the mean of the column.
    """
    if not os.path.exists(schema_path):
        print(f"Warning: Schema file not found at {schema_path}. Skipping validation.")
        return df

    with open(schema_path, 'r') as f:
        schema = yaml.safe_load(f)

    df_validated = df.copy()

    for col_name, rules in schema.items():
        if col_name not in df_validated.columns:
            print(f"Warning: Column '{col_name}' not found in DataFrame. Skipping.")
            continue

        if rules.get('type') in ['float', 'integer']:
            # Fill missing values with the mean
            if df_validated[col_name].isnull().any():
                mean_val = df_validated[col_name].mean()
                df_validated[col_name].fillna(mean_val, inplace=True)
                print(f"Info: Filled missing values in '{col_name}' with mean ({mean_val:.2f}).")

            # Clip values to range
            if 'range' in rules:
                min_val, max_val = rules['range']
                original_min = df_validated[col_name].min()
                original_max = df_validated[col_name].max()

                if original_min < min_val or original_max > max_val:
                    df_validated[col_name] = df_validated[col_name].clip(min_val, max_val)
                    print(f"Info: Clipped column '{col_name}' to range [{min_val}, {max_val}].")

    print("Schema validation and cleaning complete.")
    return df_validated


class MaterialsProjectClient:
    """Client for Materials Project API with rate limiting and error handling."""

    def __init__(self, api_key: str = None):
        if not api_key:
            raise ValueError("Materials Project API key is required. Please provide a valid API key from https://materialsproject.org/api")
        self.api_key = api_key
        self.base_url = "https://api.materialsproject.org"
        self.last_request_time = 0
        self.rate_limit_delay = 0.2  # 200ms between requests
        self.max_retries = 3

    def validate_api_key(self) -> bool:
        """Validate the API key by making a test request."""
        try:
            # Make a minimal test request
            response = self._make_request("/materials/summary/", {"_limit": 1})

            # Validate that we actually got data back
            if isinstance(response, dict) and 'data' in response:
                data = response['data']
                if isinstance(data, list) and len(data) >= 0:  # At least empty list means API is working
                    return True

            # If we get here, the response format is unexpected
            print("Warning: Unexpected API response format during validation")
            return False

        except ValueError as e:
            if "API key authentication failed" in str(e):
                return False
            raise
        except Exception:
            # Other errors (network, etc.) don't necessarily mean invalid key
            return True

    def _rate_limit_wait(self):
        """Implement rate limiting between API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a request to the Materials Project API with error handling."""
        url = f"{self.base_url}{endpoint}"

        # Ensure params exists
        if params is None:
            params = {}

        headers = {"X-API-Key": self.api_key}

        for attempt in range(self.max_retries):
            try:
                self._rate_limit_wait()

                response = requests.get(url, params=params, headers=headers, timeout=30)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    print(f"Rate limited, waiting 5 seconds (attempt {attempt + 1})")
                    time.sleep(5)
                    continue
                elif response.status_code == 403:
                    raise ValueError("API key authentication failed")
                else:
                    print(f"API error {response.status_code}: {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    raise Exception(f"API request failed after {self.max_retries} attempts")

            except requests.exceptions.RequestException as e:
                print(f"Network error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

        raise Exception("All API request attempts failed")

    def get_materials_summary(self,
                            elements: List[str] = None,
                            band_gap_min: float = None,
                            band_gap_max: float = None,
                            energy_above_hull_max: float = 0.1,
                            nsites_max: int = 20,
                            limit: int = 100) -> pd.DataFrame:
        """
        Query materials summary data with optional filters.

        Parameters:
        - elements: List of element symbols to include
        - band_gap_min/max: Band gap range in eV
        - energy_above_hull_max: Maximum energy above convex hull (stability)
        - nsites_max: Maximum number of sites in unit cell
        - limit: Maximum number of results to return
        """

        params = {
            "_fields": "material_id,formula_pretty,elements,nsites,"
                      "volume,density,density_atomic,"
                      "band_gap,energy_above_hull,"
                      "formation_energy_per_atom,total_magnetization",
            "_limit": limit
        }

        # Add filters
        if elements:
            params["elements"] = ",".join(elements)

        if band_gap_min is not None:
            params["band_gap_min"] = band_gap_min

        if band_gap_max is not None:
            params["band_gap_max"] = band_gap_max

        if energy_above_hull_max is not None:
            params["energy_above_hull_max"] = energy_above_hull_max

        if nsites_max is not None:
            params["nsites_max"] = nsites_max

        print(f"Querying Materials Project for materials with filters: {params}")

        try:
            response = self._make_request("/materials/summary/", params)
            materials = response.get("data", [])

            if not materials:
                print("No materials found matching criteria")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(materials)

            # Clean up column names and data types
            df = df.rename(columns={
                'formula_pretty': 'formula',
                'density': 'density',
                'density_atomic': 'atomic_density'
            })

            # Ensure numeric columns
            numeric_cols = ['nsites', 'volume', 'density', 'atomic_density',
                          'band_gap', 'energy_above_hull', 'formation_energy_per_atom',
                          'total_magnetization']

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            print(f"Retrieved {len(df)} materials from Materials Project")
            return df

        except Exception as e:
            print(f"Error querying materials: {e}")
            return pd.DataFrame()

    def get_binary_alloys(self,
                         element_pairs: List[Tuple[str, str]] = None,
                         limit_per_pair: int = 50) -> pd.DataFrame:
        """
        Get binary alloy materials for specified element pairs.
        This is useful for creating a balanced dataset for ML training.
        """

        if element_pairs is None:
            # Default common binary alloy systems
            element_pairs = [
                ('Al', 'Ti'), ('Al', 'V'), ('Al', 'Cr'), ('Al', 'Fe'), ('Al', 'Ni'), ('Al', 'Cu'),
                ('Ti', 'V'), ('Ti', 'Cr'), ('Ti', 'Fe'), ('Ti', 'Ni'),
                ('V', 'Cr'), ('Fe', 'Co'), ('Fe', 'Ni'), ('Co', 'Ni'), ('Ni', 'Cu')
            ]

        all_materials = []

        for elem1, elem2 in element_pairs:
            print(f"Querying binary alloys: {elem1}-{elem2}")

            materials = self.get_materials_summary(
                elements=[elem1, elem2],
                energy_above_hull_max=0.05,  # More stable materials
                nsites_max=10,
                limit=limit_per_pair
            )

            if not materials.empty:
                materials['element_1'] = elem1
                materials['element_2'] = elem2
                materials['alloy_type'] = 'binary'
                all_materials.append(materials)

            # Small delay between queries
            time.sleep(0.5)

        if not all_materials:
            print("No binary alloy materials found")
            return pd.DataFrame()

        combined_df = pd.concat(all_materials, ignore_index=True)

        # Remove duplicates based on material_id
        combined_df = combined_df.drop_duplicates(subset='material_id')

        print(f"Total binary alloys collected: {len(combined_df)}")
        return combined_df

    def get_binary_alloys_for_training(self,
                         element_pairs: List[Tuple[str, str]] = None,
                         limit_per_pair: int = 50,
                         energy_above_hull_max: float = 0.5) -> pd.DataFrame:
        """
        Get binary alloy materials for training with broader stability range.
        This includes both stable and unstable materials for balanced ML training.

        Args:
            element_pairs: List of (elem1, elem2) tuples
            limit_per_pair: Max materials per element pair
            energy_above_hull_max: Maximum E_hull to include (default 0.5 for training)
        """

        if element_pairs is None:
            # Default common binary alloy systems
            element_pairs = [
                ('Al', 'Ti'), ('Al', 'V'), ('Al', 'Cr'), ('Al', 'Fe'), ('Al', 'Ni'), ('Al', 'Cu'),
                ('Ti', 'V'), ('Ti', 'Cr'), ('Ti', 'Fe'), ('Ti', 'Ni'),
                ('V', 'Cr'), ('Fe', 'Co'), ('Fe', 'Ni'), ('Co', 'Ni'), ('Ni', 'Cu')
            ]

        all_materials = []

        for elem1, elem2 in element_pairs:
            print(f"Querying binary alloys for training: {elem1}-{elem2} (E_hull ≤ {energy_above_hull_max})")

            materials = self.get_materials_summary(
                elements=[elem1, elem2],
                energy_above_hull_max=energy_above_hull_max,  # Broader range for training
                nsites_max=10,
                limit=limit_per_pair
            )

            if not materials.empty:
                materials['element_1'] = elem1
                materials['element_2'] = elem2
                materials['alloy_type'] = 'binary'
                all_materials.append(materials)

            # Small delay between queries
            time.sleep(0.5)

        if not all_materials:
            print("No binary alloy materials found for training")
            return pd.DataFrame()

        combined_df = pd.concat(all_materials, ignore_index=True)

        # Remove duplicates based on material_id
        combined_df = combined_df.drop_duplicates(subset='material_id')

        print(f"Total binary alloys collected for training: {len(combined_df)}")
        return combined_df

    def get_material_properties(self, material_id: str) -> Dict:
        """Get detailed properties for a specific material."""
        try:
            response = self._make_request(f"/materials/{material_id}/summary/")
            return response.get("data", {})
        except Exception as e:
            print(f"Error getting properties for {material_id}: {e}")
            return {}

    def get_thermo_data(self, material_id: str) -> Dict:
        """Get thermodynamic data for a material."""
        try:
            response = self._make_request(f"/materials/{material_id}/thermo/")
            return response.get("data", {})
        except Exception as e:
            print(f"Error getting thermo data for {material_id}: {e}")
            return {}

    def get_elasticity_data(self, material_id: str) -> Dict:
        """Get elasticity data for a material."""
        try:
            response = self._make_request(f"/materials/{material_id}/elasticity/")
            return response.get("data", {})
        except Exception as e:
            print(f"Error getting elasticity data for {material_id}: {e}")
            return {}


def create_ml_features_from_mp_data(mp_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Materials Project data to ML-ready features.

    This function extracts and engineers features suitable for VAE training,
    focusing on composition, structure, and properties.
    """

    if mp_data.empty:
        return pd.DataFrame()

    # Start with basic features
    features_df = mp_data.copy()

    # Extract element properties using pymatgen

    for idx, row in features_df.iterrows():
        try:
            # Get element information
            if 'elements' in row and row['elements']:
                elements = row['elements']

                # Calculate average atomic properties
                atomic_numbers = []
                electronegativities = []
                atomic_radii = []

                for elem_symbol in elements:
                    try:
                        elem = mg.Element(elem_symbol)
                        atomic_numbers.append(elem.Z)  # Use Z instead of atomic_number

                        # Get electronegativity (Pauling scale)
                        if hasattr(elem, 'X') and elem.X is not None:
                            electronegativities.append(elem.X)

                        # Get atomic radius
                        if hasattr(elem, 'atomic_radius_calculated') and elem.atomic_radius_calculated is not None:
                            atomic_radii.append(elem.atomic_radius_calculated)
                        elif hasattr(elem, 'atomic_radius') and elem.atomic_radius is not None:
                            atomic_radii.append(elem.atomic_radius)

                    except Exception as e:
                        print(f"Warning: Could not get properties for {elem_symbol}: {e}")

                # Calculate averages
                features_df.loc[idx, 'avg_atomic_number'] = np.mean(atomic_numbers) if atomic_numbers else 0
                features_df.loc[idx, 'electronegativity_avg'] = np.mean(electronegativities) if electronegativities else 0
                features_df.loc[idx, 'atomic_radius_avg'] = np.mean(atomic_radii) if atomic_radii else 0

        except Exception as e:
            print(f"Warning: Could not process elements for material {row.get('material_id', idx)}: {e}")
            features_df.loc[idx, 'avg_atomic_number'] = 0
            features_df.loc[idx, 'electronegativity_avg'] = 0
            features_df.loc[idx, 'atomic_radius_avg'] = 0

    # Create composition features (simplified for binary alloys)
    # For binary alloys, we can estimate compositions
    features_df['composition_1'] = 0.5  # Default 50-50 for binary
    features_df['composition_2'] = 0.5
    features_df['composition_3'] = 0.0

    # Map properties to ML features
    ml_features = features_df[[
        'composition_1', 'composition_2', 'composition_3',
        'density', 'electronegativity_avg', 'atomic_radius_avg',
        'band_gap', 'energy_above_hull', 'formation_energy_per_atom'
    ]].copy()

    # Handle missing values, now done in validation
    # ml_features = ml_features.fillna(ml_features.mean())

    # Rename columns to match expected format
    ml_features = ml_features.rename(columns={
        'electronegativity_avg': 'electronegativity',
        'atomic_radius_avg': 'atomic_radius'
    })

    # Validate and clean data using the schema
    schema_path = os.path.join(os.path.dirname(__file__), '..', 'feature_schema.yml')
    if not os.path.exists(schema_path):
        # Fallback for different project structures
        schema_path = 'feature_schema.yml'
        
    ml_features = validate_data_with_schema(ml_features, schema_path)

    # Validate field consistency and standardize column names
    try:
        # Check if required fields are present
        missing_fields = [field for field in REQUIRED_FIELDS_ML_CLASSIFIER if field not in ml_features.columns]
        if missing_fields:
            print(f"Warning: Missing required fields for ML classifier: {missing_fields}")
            # Add missing fields with default values
            for field in missing_fields:
                ml_features[field] = 0.0
                print(f"Added missing field '{field}' with default value 0.0")

        # Standardize the dataframe to ensure consistent field names
        ml_features = standardize_dataframe(ml_features)
        
        print(f"Standardized ML features for {len(ml_features)} materials")
        
    except Exception as e:
        print(f"Warning: Field validation failed: {e}")
        print("Continuing with available fields...")

    return ml_features


# Convenience functions for common queries

def get_training_dataset(api_key: str = None,
                        n_materials: int = 500,
                        energy_above_hull_max: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get a balanced training dataset from Materials Project.

    Args:
        api_key: MP API key
        n_materials: Target number of materials
        energy_above_hull_max: Maximum E_hull to include (default 0.5 for training to get negatives)

    Returns:
    - ml_features: DataFrame with ML-ready features
    - raw_data: DataFrame with original Materials Project data
    """

    client = MaterialsProjectClient(api_key)

    print(f"Fetching materials from Materials Project (E_hull ≤ {energy_above_hull_max})...")

    # Get binary alloys with broader stability range for training
    binary_data = client.get_binary_alloys_for_training(
        limit_per_pair=max(20, n_materials // 15),
        energy_above_hull_max=energy_above_hull_max
    )

    if binary_data.empty:
        print("No data retrieved from Materials Project")
        return pd.DataFrame(), pd.DataFrame()

    # Create ML features
    ml_features = create_ml_features_from_mp_data(binary_data)

    print(f"Dataset ready: {len(ml_features)} materials with {len(ml_features.columns)} features")

    return ml_features, binary_data


if __name__ == "__main__":
    # Test the API integration
    print("Testing Materials Project API integration...")
    print("Note: This test requires a valid Materials Project API key.")
    print("Please provide your API key from https://materialsproject.org/api")

    # For testing, you can set your API key here or pass it as an environment variable
    test_api_key = os.getenv("MP_API_KEY")
    if not test_api_key:
        print("No API key found. Set MP_API_KEY environment variable or modify the test code.")
        exit(1)

    try:
        client = MaterialsProjectClient(test_api_key)

        # Test basic query
        test_data = client.get_materials_summary(
            elements=["Al", "Ti"],
            energy_above_hull_max=0.05,
            limit=5
        )

        if not test_data.empty:
            print("API connection successful!")
            print(test_data[['material_id', 'formula', 'band_gap', 'density']].head())
        else:
            print("API connection failed or no data returned")
    except ValueError as e:
        print(f"API key validation failed: {e}")
    except Exception as e:
        print(f"Test failed: {e}")
