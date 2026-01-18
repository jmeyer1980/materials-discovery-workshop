"""
Materials Dataset Generator for ML-based Material Discovery

This script generates synthetic datasets of alloy compositions with
associated material properties for training ML models.

DATA SOURCES:
- Element properties obtained from Pymatgen library:
  Ong, S. P., Richards, W. D., Jain, A., Hautier, G., Kocher, M., Cholia, S.,
  ... & Ceder, G. (2013). "Python Materials Genomics (pymatgen): A robust,
  open-source python library for materials analysis."
  Computational Materials Science, 68, 314-319.
  doi: 10.1016/j.commatsci.2012.10.028

METHODOLOGY:
- Properties calculated using Rule of Mixtures (standard materials science)
- Synthetic data generated for demonstration purposes
- Not derived from real experimental measurements

DEPENDENCIES:
- Pymatgen: Element property data
- NumPy: Numerical operations
- Pandas: Data structure management

LICENSE: MIT (see LICENSE file in repository)
"""

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element
from typing import List, Dict, Tuple
import random


class MaterialsDatasetGenerator:
    """Generator for synthetic materials dataset."""

    def __init__(self, seed: int = 42):
        """Initialize the dataset generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # Common alloy elements (excluding noble gases, radioactive elements)
        self.elements = [
            'Al', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf',
            'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Mg', 'Ca',
            'Sr', 'Ba', 'Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu',
            'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'
        ]

        # Element properties for feature engineering
        self.element_properties = self._load_element_properties()

    def _load_element_properties(self) -> Dict[str, Dict[str, float]]:
        """Load basic element properties for feature engineering."""
        properties = {}

        for elem_symbol in self.elements:
            try:
                elem = Element(elem_symbol)
                properties[elem_symbol] = {
                    'atomic_number': elem.Z,
                    'atomic_mass': elem.atomic_mass,
                    'electronegativity': elem.X if elem.X else 0.0,
                    'atomic_radius': elem.atomic_radius if elem.atomic_radius else 0.0,
                    'melting_point': elem.melting_point if elem.melting_point else 0.0,
                    'boiling_point': elem.boiling_point if elem.boiling_point else 0.0,
                    'density': elem.density_of_solid if elem.density_of_solid else 0.0,
                }
            except:
                # Fallback values for elements with missing data
                properties[elem_symbol] = {
                    'atomic_number': 0,
                    'atomic_mass': 0,
                    'electronegativity': 0,
                    'atomic_radius': 0,
                    'melting_point': 0,
                    'boiling_point': 0,
                    'density': 0,
                }

        return properties

    def generate_binary_alloys(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate binary alloy compositions.

        Args:
            n_samples: Number of alloy samples to generate

        Returns:
            DataFrame with alloy compositions and properties
        """
        data = []

        for _ in range(n_samples):
            # Randomly select two different elements
            elem1, elem2 = random.sample(self.elements, 2)

            # Generate random composition (atomic percentages)
            comp1 = random.uniform(0.1, 0.9)  # 10% to 90%
            comp2 = 1.0 - comp1

            # Create composition string
            formula = f"{elem1}{comp1:.3f}{elem2}{comp2:.3f}"

            # Calculate properties
            properties = self._calculate_alloy_properties([(elem1, comp1), (elem2, comp2)])

            data.append({
                'formula': formula,
                'element_1': elem1,
                'element_2': elem2,
                'composition_1': comp1,
                'composition_2': comp2,
                **properties
            })

        return pd.DataFrame(data)

    def generate_ternary_alloys(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate ternary alloy compositions.

        Args:
            n_samples: Number of alloy samples to generate

        Returns:
            DataFrame with alloy compositions and properties
        """
        data = []

        for _ in range(n_samples):
            # Randomly select three different elements
            elem1, elem2, elem3 = random.sample(self.elements, 3)

            # Generate random composition (must sum to 1)
            comp1 = random.uniform(0.1, 0.6)
            comp2 = random.uniform(0.1, 0.6)
            comp3 = 1.0 - comp1 - comp2

            # Ensure all compositions are positive
            if comp3 <= 0.05:
                comp1 = comp1 / (comp1 + comp2) * 0.95
                comp2 = comp2 / (comp1 + comp2) * 0.95
                comp3 = 0.05

            # Create composition string
            formula = f"{elem1}{comp1:.3f}{elem2}{comp2:.3f}{elem3}{comp3:.3f}"

            # Calculate properties
            properties = self._calculate_alloy_properties([
                (elem1, comp1), (elem2, comp2), (elem3, comp3)
            ])

            data.append({
                'formula': formula,
                'element_1': elem1,
                'element_2': elem2,
                'element_3': elem3,
                'composition_1': comp1,
                'composition_2': comp2,
                'composition_3': comp3,
                **properties
            })

        return pd.DataFrame(data)

    def _calculate_alloy_properties(self, composition: List[Tuple[str, float]]) -> Dict[str, float]:
        """Calculate alloy properties using rule of mixtures.

        Args:
            composition: List of (element, fraction) tuples

        Returns:
            Dictionary of calculated properties
        """
        # Initialize weighted sums
        total_mass = 0
        weighted_mp = 0
        weighted_bp = 0
        weighted_density = 0
        avg_electronegativity = 0
        avg_atomic_radius = 0

        for elem, frac in composition:
            props = self.element_properties[elem]

            # Calculate weighted properties
            weighted_mp += props['melting_point'] * frac
            weighted_bp += props['boiling_point'] * frac
            weighted_density += props['density'] * frac
            avg_electronegativity += props['electronegativity'] * frac
            avg_atomic_radius += props['atomic_radius'] * frac

        # Calculate electronegativity difference (for binary alloys)
        if len(composition) == 2:
            elem1, elem2 = composition[0][0], composition[1][0]
            delta_chi = abs(self.element_properties[elem1]['electronegativity'] -
                           self.element_properties[elem2]['electronegativity'])
        else:
            delta_chi = 0  # For ternary, use average difference or set to 0

        # Add some noise to simulate real data variability
        noise_level = 0.1

        return {
            'melting_point': weighted_mp * (1 + np.random.normal(0, noise_level)),
            'boiling_point': weighted_bp * (1 + np.random.normal(0, noise_level)),
            'density': weighted_density * (1 + np.random.normal(0, noise_level)),
            'electronegativity': avg_electronegativity,
            'atomic_radius': avg_atomic_radius,
            'electronegativity_difference': delta_chi,
        }

    def generate_dataset(self, n_binary: int = 2000, n_ternary: int = 1000) -> pd.DataFrame:
        """Generate complete dataset with binary and ternary alloys.

        Args:
            n_binary: Number of binary alloy samples
            n_ternary: Number of ternary alloy samples

        Returns:
            Combined DataFrame with all alloy data
        """
        print(f"Generating {n_binary} binary alloys...")
        binary_data = self.generate_binary_alloys(n_binary)

        print(f"Generating {n_ternary} ternary alloys...")
        ternary_data = self.generate_ternary_alloys(n_ternary)

        # Combine datasets
        combined_data = pd.concat([binary_data, ternary_data], ignore_index=True)

        # Add alloy type column
        combined_data['alloy_type'] = ['binary'] * len(binary_data) + ['ternary'] * len(ternary_data)

        return combined_data


def main():
    """Generate and save the materials dataset."""
    generator = MaterialsDatasetGenerator(seed=42)

    # Generate dataset
    dataset = generator.generate_dataset(n_binary=2000, n_ternary=1000)

    # Save to CSV
    dataset.to_csv('materials_dataset.csv', index=False)

    print(f"Generated dataset with {len(dataset)} samples")
    print("Columns:", list(dataset.columns))
    print("\nSample data:")
    print(dataset.head())

    # Basic statistics
    print(f"\nDataset statistics:")
    print(f"Binary alloys: {len(dataset[dataset['alloy_type'] == 'binary'])}")
    print(f"Ternary alloys: {len(dataset[dataset['alloy_type'] == 'ternary'])}")
    print(f"Melting point range: {dataset['melting_point'].min():.1f} - {dataset['melting_point'].max():.1f} K")
    print(f"Density range: {dataset['density'].min():.3f} - {dataset['density'].max():.3f} g/cm³")


if __name__ == "__main__":
    main()
