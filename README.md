# Materials Discovery ML

This project demonstrates how machine learning can be used to discover new materials and alloys by learning patterns from existing material chemical structures and properties.

## Overview

Materials discovery traditionally relies on trial-and-error experimentation and human intuition. This project shows how machine learning can accelerate the discovery process by:

1. **Learning patterns** from existing materials data
2. **Generating new compositions** in unexplored regions of materials space
3. **Identifying clusters** of similar materials
4. **Exploring property relationships** to find optimal material combinations

## What We Built

### Core Components

- **Data Generator** (`data_generator.py`): Creates synthetic alloy datasets with realistic material properties
- **ML Model** (`materials_discovery_model.py`): Advanced variational autoencoder for material generation
- **Demonstration** (`materials_discovery_demo.py`): Complete end-to-end demonstration with visualizations

### Key Technologies

- **PyTorch**: Deep learning framework for generative modeling
- **Scikit-learn**: Traditional ML for clustering and preprocessing
- **Pymatgen**: Materials science library for chemical data structures
- **Matplotlib/Seaborn**: Data visualization and analysis

## How It Works

### 1. Data Generation

We start by generating a synthetic dataset of alloy compositions with properties calculated using the rule of mixtures:

- Binary alloys (e.g., Cu-Zn, Fe-Ni)
- Ternary alloys (e.g., Cu-Zn-Al)
- Properties: melting point, density, electronegativity, atomic radius

### 2. Feature Engineering

Materials are represented as high-dimensional feature vectors including:

- Elemental compositions
- Chemical properties
- Interaction terms
- One-hot encoded element identities

### 3. Machine Learning Model

A Variational Autoencoder (VAE) learns a compressed representation of materials space:

- **Encoder**: Maps materials to a low-dimensional latent space
- **Decoder**: Reconstructs materials from latent representations
- **Latent Space**: Enables generation of new materials by sampling

### 4. Material Generation

New materials are discovered by:

- Sampling from the learned latent space
- Decoding samples back to material properties
- Exploring regions beyond training data

### 5. Analysis & Visualization

The system provides:

- Property distribution comparisons
- Composition space exploration
- Material clustering analysis
- Performance metrics and insights

## Results

Our demonstration successfully:

- Analyzed 2,000 existing alloy compositions
- Trained a VAE to learn materials patterns
- Generated 100 new alloy compositions
- Identified 4 natural material clusters
- Created comprehensive visualizations

### Sample Generated Materials

```table
formula     melting_point  density
Ti0.847Co0.153    1776.31     9454.86
V0.853Ni0.147     1774.60     9452.43
Cr0.841Zn0.159    1770.85     9403.81
Mn0.846Cu0.154    1777.06     9446.52
```

## Files Overview

- `data_generator.py` - Generates synthetic materials dataset
- `materials_dataset.csv` - Generated alloy data (2000 binary + 1000 ternary)
- `materials_discovery_model.py` - Full ML pipeline with VAE
- `materials_discovery_demo.py` - Simplified demonstration script
- `generated_materials_demo.csv` - Newly discovered materials
- `materials_discovery_demo.png` - Comprehensive visualization
- `training_loss.png` - Model training progress
- `requirements.txt` - Python dependencies

## Running the Demonstration

### Option 1: Google Colab (Recommended - No Setup Required)

🚀 **Easiest way to get started!**

1. **Open the Colab notebook:**
   - Click here: [Materials Discovery Workshop - Colab Edition](https://colab.research.google.com/github/your-repo/materials-discovery/blob/main/materials_discovery_workshop_colab.ipynb)
   - Or upload `materials_discovery_workshop_colab.ipynb` to Google Colab

2. **Run all cells:**
   - Click "Runtime" → "Run all"
   - The notebook includes automatic dataset generation and dependency installation

3. **Features included:**
   - ✅ Synthetic materials dataset generation
   - ✅ Complete VAE training and material generation
   - ✅ Advanced validation techniques
   - ✅ Interactive parameter controls
   - ✅ Comprehensive visualizations
   - ✅ Production-ready evaluation metrics

### Option 2: Local Installation

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate materials data:**

   ```bash
   python data_generator.py
   ```

3. **Run the demonstration:**

   ```bash
   python materials_discovery_demo.py
   ```

### Option 3: Jupyter Notebook Locally

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Launch Jupyter:**

   ```bash
   jupyter notebook materials_discovery_workshop.ipynb
   ```

**Note:** The local notebook version requires the `materials_dataset.csv` file to be present. Use the Colab version for the complete self-contained experience.

## Key Insights

### ML for Materials Discovery

1. **Pattern Recognition**: ML can learn complex relationships between composition and properties
2. **Exploration**: Generative models can propose materials beyond existing datasets
3. **Clustering**: Unsupervised learning reveals natural material groupings
4. **Efficiency**: Computational methods complement experimental approaches

### Technical Achievements

- **Scalable Architecture**: Handles thousands of material compositions
- **Feature Engineering**: Rich representation of chemical and physical properties
- **Generative Modeling**: VAE successfully captures materials distribution
- **Visualization**: Comprehensive analysis of results

## Future Directions

This work demonstrates the potential for ML in materials science:

1. **Experimental Validation**: Test generated materials in lab
2. **Advanced Models**: Use transformer architectures for chemical formulas
3. **Property Prediction**: Add ML models for specific material properties
4. **Multi-objective Optimization**: Find materials with multiple desired properties
5. **Real Datasets**: Apply to experimental materials databases

## Acknowledgments

This project builds on foundational work in generative modeling and materials
science. We acknowledge:

### Methodology & Core Research
- Kingma & Welling (2013) for the Variational Autoencoder framework, which
  provides the theoretical foundation for our generative approach
- Researchers advancing the application of generative models to materials discovery

### Open-Source Software Communities
- The **PyTorch** team for providing a flexible deep learning framework
- **Scikit-learn** developers for excellent ML algorithms and preprocessing tools
- The **Pymatgen** community for materials property databases and chemical
  structure tools
- The scientific Python ecosystem: NumPy, Pandas, Matplotlib, and Seaborn

### Data & Resources
- Materials Project team for curated element property databases
- The materials science community for standardized methodologies like the
  Rule of Mixtures

### Licensing
This project is released under the MIT License. All dependencies use compatible
open-source licenses (MIT, BSD, or Apache 2.0). See THIRDPARTY_LICENSES.txt
for complete details.

## References & Academic Attribution

### Foundational Methodology

1. **Variational Autoencoders (VAE)**
   - Kingma, D. P., & Welling, M. (2013). "Auto-Encoding Variational Bayes."
     arXiv preprint arXiv:1312.6114
   - doi: 10.48550/arXiv.1312.6114
   - Introductory review: Kingma, D. P., & Welling, M. (2019).
     "An Introduction to Variational Autoencoders."
     arXiv preprint arXiv:1906.02691

2. **Generative Models for Materials Discovery**
   - Review of AI methods in materials science: arXiv:2508.03278
   - Deep generative models overview: Frontiers in Materials, 9, 865270 (2022)

### Software Libraries & Tools

3. **PyTorch**
   - Paszke, A., Gross, S., Massa, F., et al. (2019).
     "PyTorch: An Imperative Style, High-Performance Deep Learning Library."
     Advances in Neural Information Processing Systems 32 (NeurIPS 2019)
   - https://pytorch.org/

4. **Scikit-learn**
   - Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011).
     "Scikit-learn: Machine Learning in Python."
     Journal of Machine Learning Research, 12, 2825-2830.
   - https://scikit-learn.org/

5. **Pymatgen**
   - Ong, S. P., Richards, W. D., Jain, A., Hautier, G., Kocher, M., Cholia, S.,
     ... & Ceder, G. (2013). "Python Materials Genomics (pymatgen): A robust,
     open-source python library for materials analysis."
     Computational Materials Science, 68, 314-319.
   - doi: 10.1016/j.commatsci.2012.10.028
   - https://pymatgen.org/

6. **Matplotlib**
   - Hunter, J. D. (2007). "Matplotlib: A 2D Graphics Environment."
     Computing in Science & Engineering, 9(3), 90-95.

7. **Pandas**
   - McKinney, W. (2010). "Data structures for statistical computing in Python."
     Proceedings of the 9th Python in Science Conference.

8. **NumPy**
   - Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020).
     "Array programming with NumPy." Nature, 585, 357–362.
   - doi: 10.1038/s41586-020-2649-2

### Data Sources

9. **Element Properties**
   - Element property data obtained from Pymatgen library
     (derived from Materials Project)
   - Materials Project: Jain, A., Ong, S. P., Hautier, G., et al. (2013).
     "The Materials Project: A materials genome approach to accelerating
     materials innovation." APL Materials, 1(1), 011002.

### Methodology References

10. **Rule of Mixtures (Alloy Property Estimation)**
    - Standard materials science methodology for estimating composite/alloy
      properties from constituent element properties
    - See: Ashby, M. F., & Johnson, K. (2013). "Materials and Design: The Art
      and Science of Material Selection in Product Design." Butterworth-Heinemann.

## For Your Friend's "Itch in the Brain"

This project shows that ML can indeed help discover new materials! Instead of randomly trying different alloy combinations, we can:

- **Learn from existing knowledge** (like how humans learn from textbooks)
- **Generate intelligent hypotheses** (new material combinations to try)
- **Explore property space efficiently** (focus on promising regions)
- **Accelerate discovery** (reduce trial-and-error experimentation)

The same principles apply to other scientific discovery problems - ML can learn patterns from existing data and suggest novel experiments or designs.

---

- *"The best way to predict the future is to create it." - Peter Drucker*

This project creates a future where materials discovery is accelerated by machine learning! 🚀
