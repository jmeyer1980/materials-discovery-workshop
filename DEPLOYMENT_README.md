# Materials Discovery ML - Workshop Deployment Guide

This guide provides multiple deployment options for the Materials Discovery ML system in fabrication workshop environments.

## 📋 Prerequisites

- Python 3.8+ (for local deployment)
- Docker & Docker Compose (for containerized deployment)
- Git (to clone/download the project)
- 4GB+ RAM recommended for ML training
- Internet connection for initial setup

## 🚀 Quick Start Options

### Option 1: Jupyter Notebook Workshop (Interactive Learning)

Perfect for educational workshops where users need to understand the ML process step-by-step.

**Requirements:** Python environment with required packages

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook materials_discovery_workshop.ipynb
```

**Features:**
- Step-by-step ML workflow explanation
- Interactive parameter adjustment
- Real-time visualization generation
- Educational content for materials scientists

### Option 2: Streamlit Web App (Production Interface)

Simple web interface for generating new materials on-demand.

**Local Deployment:**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

**Features:**
- One-click material generation
- Adjustable sample count
- Real-time property visualization
- Export generated materials to CSV

### Option 3: Docker Container (Portable Deployment)

Fully containerized solution for easy deployment across different systems.

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or with Docker directly
docker build -t materials-discovery .
docker run -p 8501:8501 materials-discovery
```

Open http://localhost:8501 in your browser.

**Benefits:**
- No dependency conflicts
- Consistent environment across machines
- Easy scaling and distribution
- Includes all required data and models

## 📁 Project Structure

```
materials-discovery-workshop/
├── materials_discovery_workshop.ipynb  # Interactive workshop notebook
├── app.py                              # Streamlit web application
├── materials_discovery_demo.py         # Original demo script
├── materials_dataset.csv              # Training data (3000 alloys)
├── generated_materials_demo.csv       # Example generated materials
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker container definition
├── docker-compose.yml                 # Docker Compose configuration
├── data_generator.py                  # Synthetic data generation
├── materials_discovery_model.py       # ML model definitions
└── DEPLOYMENT_README.md              # This file
```

## 🔧 Detailed Setup Instructions

### Python Environment Setup

1. **Create Virtual Environment (Recommended):**
   ```bash
   python -m venv materials_env
   source materials_env/bin/activate  # Linux/Mac
   # or
   materials_env\Scripts\activate     # Windows
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation:**
   ```bash
   python -c "import torch, pandas, streamlit, sklearn; print('All dependencies installed!')"
   ```

### Data Requirements

The system comes with pre-generated training data (`materials_dataset.csv`). If you need to regenerate or modify the dataset:

```bash
python data_generator.py
```

This will create a fresh dataset with customizable alloy compositions.

## 🎯 Usage Instructions

### For Workshop Facilitators

1. **Prepare the Environment:**
   - Set up Python environment or Docker container
   - Ensure all participants have access to the notebook/app

2. **Workshop Flow:**
   - Start with Jupyter notebook for theory and hands-on learning
   - Demonstrate the Streamlit app for practical generation
   - Discuss real-world applications and limitations

3. **Key Discussion Points:**
   - How ML learns patterns from existing materials
   - Importance of experimental validation
   - Potential for accelerating R&D workflows

### For Fabrication Teams

1. **Daily Usage:**
   - Launch the Streamlit app: `streamlit run app.py`
   - Generate candidate materials for testing
   - Export results for experimental planning

2. **Integration with Existing Workflows:**
   - Generated materials can be imported into CAD/CAM systems
   - Property predictions can guide initial screening
   - Clustering results help identify material families

## 🔍 Troubleshooting

### Common Issues

**"Module not found" errors:**
- Ensure you're in the correct virtual environment
- Run `pip install -r requirements.txt` again
- For Docker: rebuild with `docker-compose build --no-cache`

**Streamlit app not loading:**
- Check that port 8501 is available
- Try different port: `streamlit run app.py --server.port=8502`
- For Docker: ensure port mapping is correct

**ML training taking too long:**
- Reduce epochs in the code (default is optimized for speed)
- Use GPU if available (automatically detected)
- For workshops: pre-train models and load from checkpoint

**Memory errors:**
- Reduce batch size in training loops
- Generate fewer samples at once
- Use lighter model configurations

### Performance Optimization

**For Low-Resource Systems:**
- Reduce latent dimension from 5 to 3 in VAE
- Decrease training epochs to 20-30
- Generate smaller batches of materials (10-20 at a time)

**For High-Performance Systems:**
- Increase latent dimension to 8-10
- Train for more epochs (100+)
- Use larger datasets and more complex models

## 📊 Understanding the Results

### Generated Materials Interpretation

- **Novelty:** Materials are generated in unexplored regions of property space
- **Realism:** Based on learned patterns from existing alloys
- **Validation Required:** Always test experimentally before production use

### Clustering Insights

- **Material Families:** Groups of similar alloys with shared properties
- **Design Guidelines:** Use clusters to understand property relationships
- **Optimization Targets:** Focus experimental efforts on promising clusters

## 🔄 Updating and Maintenance

### Adding New Data

1. Generate or obtain new alloy data
2. Format as CSV with required columns
3. Retrain models with updated dataset
4. Update the Streamlit app to use new models

### Model Improvements

- Experiment with different VAE architectures
- Add more material properties to features
- Implement conditional generation (target specific properties)
- Add reinforcement learning for property optimization

## 🤝 Contributing and Support

### For Workshop Customization

- Modify the Jupyter notebook for your specific materials focus
- Add domain-specific visualizations
- Include case studies from your fabrication experience
- Integrate with existing lab equipment APIs

### Getting Help

- Check the troubleshooting section above
- Review the original demo script for implementation details
- Examine the generated visualizations for insight into model behavior
- Validate results experimentally before trusting predictions

## 📈 Next Steps and Advanced Usage

### Research Directions

- **Multi-Objective Optimization:** Generate materials with target property ranges
- **Ternary Alloys:** Extend to 3+ element systems
- **Property Prediction:** Add ML models for predicting additional properties
- **Active Learning:** Iteratively improve models with experimental feedback

### Production Integration

- **Database Integration:** Store generated materials in lab databases
- **Workflow Automation:** Connect to experimental planning systems
- **Quality Control:** Implement validation pipelines for generated materials
- **Scalability:** Deploy on cloud infrastructure for team access

---

**Remember:** This tool accelerates materials discovery but cannot replace experimental validation. Always test ML-generated candidates thoroughly before fabrication and use.

For questions or customizations, refer to the code comments and documentation in the source files.
