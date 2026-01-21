# User Guide: Materials Discovery Workshop

## Table of Contents
1. [Quick Start](#quick-start)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [Understanding Results](#understanding-results)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Quick Start

### Option 1: Google Colab (Recommended)
🚀 **No installation required!**

1. Open [Materials Discovery Workshop Colab](https://colab.research.google.com/github/jmeyer1980/materials-discovery-workshop/blob/main/materials_discovery_workshop_colab_real_data.ipynb)
2. Click "Runtime" → "Run all"
3. Start exploring materials!

### Option 2: Local Installation
```bash
# 1. Clone and setup
git clone https://github.com/jmeyer1980/materials-discovery-workshop.git
cd materials-discovery-workshop

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get Materials Project API key
# Visit https://materialsproject.org/api and get your free API key
export MP_API_KEY="your_api_key_here"

# 4. Run the application
python gradio_app.py
```

## System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **RAM**: 4GB
- **Storage**: 500MB
- **Internet**: Required for Materials Project API access

### Recommended Requirements
- **Python**: 3.9+
- **RAM**: 8GB
- **Storage**: 1GB
- **GPU**: Optional (for faster model training)

### Supported Platforms
- **Linux**: Ubuntu 18.04+, CentOS 7+
- **macOS**: 10.15+
- **Windows**: 10+ (via WSL recommended)

## Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/jmeyer1980/materials-discovery-workshop.git
cd materials-discovery-workshop
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Get Materials Project API Key
1. Visit [Materials Project](https://materialsproject.org/)
2. Create a free account
3. Go to API section and generate your API key
4. Set environment variable:
   ```bash
   export MP_API_KEY="your_api_key_here"
   ```

### Step 5: Verify Installation
```bash
python -c "from synthesizability_predictor import SynthesizabilityClassifier; print('Installation successful!')"
```

## Basic Usage

### Starting the Application
```bash
python gradio_app.py
```

The application will start a local web server. Open the URL shown in your browser.

### Main Interface

#### 1. Materials Input
- **Upload CSV**: Upload your materials data
- **Manual Entry**: Enter individual material properties
- **Example Data**: Use provided sample materials

#### 2. Prediction Settings
- **SAFE MODE**: Enable/disable safety filtering
- **Ensemble Weights**: Balance ML vs rule-based predictions
- **Calibration**: Choose calibration method

#### 3. Results View
- **Predictions Table**: Synthesizability predictions for each material
- **Summary Statistics**: Overall statistics and confidence metrics
- **Export Options**: Download results as CSV or PDF

### Sample Workflow

1. **Upload Materials Data**
   ```
   CSV Format Required:
   formation_energy_per_atom,band_gap,energy_above_hull,electronegativity,atomic_radius,nsites,density
   -2.5,1.2,0.02,1.8,1.4,8,7.2
   ```

2. **Configure Settings**
   - SAFE MODE: On (recommended for lab use)
   - Ensemble: ML 70%, Rule-based 30%

3. **Run Predictions**
   - Click "Predict Synthesizability"
   - Wait for results (usually <30 seconds)

4. **Review Results**
   - Sort by probability to prioritize high-confidence materials
   - Check calibration status for reliability
   - Export top candidates for synthesis

## Advanced Features

### Model Calibration
The system uses probability calibration to ensure reliable confidence estimates:

```python
from synthesizability_predictor import SynthesizabilityClassifier

classifier = SynthesizabilityClassifier()
classifier.train()  # Trains base model
classifier.calibrate_model(method='isotonic')  # Adds calibration

# Get calibration metrics
calibration_info = classifier.get_calibration_summary()
print(f"ECE: {calibration_info['expected_calibration_error']:.3f}")
```

### Ensemble Weighting
Optimize the balance between ML and rule-based predictions:

```python
# Default weights
ensemble_weights = {'ml': 0.7, 'llm': 0.3}

# Reweight based on calibration performance
new_weights = classifier.reweight_ensemble()
print(f"Optimized weights: {new_weights}")
```

### Safety Filtering
Configure hazard screening for lab-ready exports:

```python
from export_for_lab import export_for_lab

# Safe mode (recommended for lab use)
csv_path, pdf_path, summary = export_for_lab(predictions_df, safe_mode=True)

# Results include safety filtering statistics
print(f"Safe materials: {summary['safe_materials']}")
print(f"Blocked materials: {summary['unsafe_materials']}")
```

### Batch Processing
Process large datasets efficiently:

```python
# Load large dataset
large_dataset = pd.read_csv('my_materials.csv')

# Process in batches
batch_size = 100
results = []

for i in range(0, len(large_dataset), batch_size):
    batch = large_dataset.iloc[i:i+batch_size]
    batch_predictions = classifier.predict(batch)
    results.append(batch_predictions)

final_results = pd.concat(results, ignore_index=True)
```

## Understanding Results

### Prediction Columns

| Column | Description | Interpretation |
|--------|-------------|----------------|
| `prediction` | Binary classification | 1 = synthesizable, 0 = not synthesizable |
| `probability` | Calibrated probability | 0.0-1.0 (higher = more likely synthesizable) |
| `confidence` | Decision certainty | 0.0-1.0 (higher = more confident prediction) |
| `calibration_status` | Probability reliability | 'well-calibrated', 'overconfident', 'underconfident' |
| `in_distribution` | Data novelty | 'in-dist' = similar to training data, 'out-dist' = novel |

### Confidence Levels

- **High Confidence** (`confidence` > 0.8): Very reliable predictions
- **Medium Confidence** (0.6 < `confidence` ≤ 0.8): Moderately reliable
- **Low Confidence** (`confidence` ≤ 0.6): Exercise caution, may need expert review

### Safety Status

- **Safe**: Passed all safety checks, ready for lab export
- **Human Review**: Requires expert approval (e.g., hazardous elements with override)
- **Blocked**: Failed safety criteria (e.g., contains prohibited elements)

## Troubleshooting

### Common Issues

#### 1. API Key Issues
```
Error: MP API key validation failed
```
**Solution:**
```bash
# Check if key is set
echo $MP_API_KEY

# Set key properly
export MP_API_KEY="your_key_here"

# Test key
python -c "from materials_discovery_api import MaterialsProjectClient; client = MaterialsProjectClient('$MP_API_KEY'); print(client.validate_api_key())"
```

#### 2. Memory Issues
```
Error: Out of memory during training
```
**Solution:**
- Reduce dataset size: `classifier.train(n_materials=100)`
- Use smaller model: `SynthesizabilityClassifier(model_type='gradient_boosting')`
- Increase system RAM or use cloud instance

#### 3. Calibration Errors
```
Warning: Calibration curve not available
```
**Solution:**
- Ensure model is trained before calibration
- Check that training data includes both classes
- Try different calibration method: `calibrate_model(method='platt')`

#### 4. Import Errors
```
ModuleNotFoundError: No module named 'pymatgen'
```
**Solution:**
```bash
pip install pymatgen
# Or for full installation:
conda install -c conda-forge pymatgen
```

### Performance Issues

#### Slow Predictions
- **Cause**: Large dataset or complex model
- **Solution**: Process in batches, use simpler model

#### Poor Accuracy
- **Cause**: Materials outside training distribution
- **Solution**: Check `in_distribution` column, consider retraining with broader data

#### Calibration Drift
- **Cause**: Model used on very different materials than training data
- **Solution**: Monitor ECE, retrain periodically with new data

## Best Practices

### Data Preparation

1. **Standardize Units**
   - Energy: eV/atom
   - Length: Å (angstroms)
   - Density: g/cm³
   - Angles: degrees

2. **Handle Missing Values**
   - Use mean imputation for missing numeric values
   - Remove materials with too many missing properties

3. **Validate Ranges**
   - Energy above hull: typically 0-2 eV/atom
   - Formation energy: typically -10 to +2 eV/atom
   - Band gap: typically 0-10 eV

### Prediction Usage

1. **Prioritize High-Confidence Predictions**
   ```python
   high_confidence = results[results['confidence'] > 0.8]
   top_candidates = high_confidence.nlargest(10, 'probability')
   ```

2. **Consider Safety First**
   ```python
   # Always use safe mode for lab exports
   export_for_lab(results, safe_mode=True)
   ```

3. **Monitor Calibration**
   ```python
   cal_info = classifier.get_calibration_summary()
   if cal_info['expected_calibration_error'] > 0.15:
       print("Warning: Model may need recalibration")
   ```

### Experimental Planning

1. **Start Small**: Begin with 3-5 high-confidence materials
2. **Document Conditions**: Record all synthesis parameters
3. **Track Outcomes**: Update local knowledge base with results
4. **Iterate**: Use results to improve future predictions

### Maintenance

1. **Regular Updates**: Retrain monthly with new MP data
2. **Monitor Performance**: Track prediction accuracy vs experimental results
3. **Version Control**: Keep track of model versions and changes
4. **Backup Data**: Save predictions and experimental outcomes

## Support and Resources

### Getting Help

- **Documentation**: See MODEL_CARD.md for technical details
- **Issues**: Report bugs on GitHub Issues
- **Discussions**: Use GitHub Discussions for questions

### Example Scripts

#### Basic Prediction Script
```python
#!/usr/bin/env python3
import pandas as pd
from synthesizability_predictor import SynthesizabilityClassifier

# Load your materials
materials = pd.read_csv('my_materials.csv')

# Initialize and train model
classifier = SynthesizabilityClassifier()
metrics = classifier.train()

# Make predictions
predictions = classifier.predict(materials)

# Save results
predictions.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
```

#### Advanced Analysis Script
```python
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from synthesizability_predictor import SynthesizabilityClassifier

# Load and predict
materials = pd.read_csv('materials.csv')
classifier = SynthesizabilityClassifier()
classifier.train()
results = classifier.predict(materials)

# Analysis
print("Summary Statistics:")
print(f"Total materials: {len(results)}")
print(f"Synthesizable: {(results['prediction'] == 1).sum()}")
print(f"Average probability: {results['probability'].mean():.3f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.hist(results['probability'], bins=20, alpha=0.7)
plt.xlabel('Synthesizability Probability')
plt.ylabel('Number of Materials')
plt.title('Distribution of Synthesizability Predictions')
plt.savefig('probability_distribution.png')
plt.show()
```

## Changelog

### Version 1.0.0 (January 2026)
- ✅ Complete rewrite for synthesizability prediction
- ✅ Real Materials Project data integration
- ✅ Advanced calibration and ensemble methods
- ✅ Safety filtering and lab-ready exports
- ✅ Comprehensive documentation and examples

### Version 0.9.0 (December 2025)
- ✅ Gradio web interface
- ✅ Basic calibration implementation
- ✅ Materials Project API integration

### Version 0.8.0 (November 2025)
- ✅ Initial ML model with Random Forest
- ✅ Basic prediction functionality
- ✅ Synthetic data validation

---

## Success Stories

*"This tool saved us months of experimental work by helping us prioritize which of our 200 DFT-predicted materials to synthesize first. We successfully synthesized 3 new alloys that showed promising magnetic properties."*

*"The calibration feature gives us confidence in the predictions. We can now focus our limited lab resources on materials with the highest chance of success."*

*"The safety filtering is a game-changer for our student lab. It prevents accidents while still allowing access to interesting materials with proper oversight."*

---

*Happy materials discovering!* 🔬✨
