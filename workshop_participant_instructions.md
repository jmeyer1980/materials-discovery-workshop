# Materials Discovery ML Workshop - Participant Instructions

## Workshop Overview

This hands-on workshop introduces machine learning techniques for materials discovery. You'll learn how to use variational autoencoders (VAEs) to generate new alloy compositions from existing materials data.

**Duration:** 2-3 hours
**Prerequisites:** Basic Python knowledge, curiosity about materials science
**Tools Needed:** Google Colab (no installation required)

## Step-by-Step Instructions

### 1. Access the Workshop Notebook

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload the `materials_discovery_workshop.ipynb` file
3. If prompted, click "Run anyway" for the notebook

### 2. Initial Setup

1. **Run the first cell** (imports and setup)
   - This installs all necessary libraries
   - Should take 1-2 minutes

2. **Set Interactive Parameters** (new feature!)
   - Use the sliders to adjust:
     - **Latent Dimension**: Model complexity (2-20, default 5)
     - **Training Epochs**: How long to train (10-200, default 50)
     - **Number of Samples**: How many materials to generate (10-500, default 100)

### 3. Data Exploration

3. **Load and explore data**
   - Examine the materials dataset
   - Understand alloy types and properties

4. **Data preprocessing**
   - Focus on binary alloys for this demo
   - Scale features for ML

### 4. Build the VAE Model

5. **Define the VAE architecture**
   - Encoder compresses data to latent space
   - Decoder reconstructs from latent space

6. **Train the model**
   - Watch the training progress
   - Monitor loss decrease over epochs

### 5. Generate New Materials

7. **Sample from latent space**
   - Generate novel alloy compositions
   - Convert to interpretable formulas

### 6. Analysis and Visualization

8. **Cluster analysis**
   - Group materials by similarity
   - Visualize material relationships

9. **Compare original vs generated**
   - Property distributions
   - Composition space analysis

### 7. Export and Save

10. **Save results**
    - Download generated materials CSV
    - Save visualization images

## Interactive Experiments

Try these parameter adjustments to see how they affect results:

### Experiment 1: Model Complexity
- **Low latent dimension** (2-3): Simpler model, faster training
- **High latent dimension** (10-15): More complex patterns, slower training

### Experiment 2: Training Duration
- **Few epochs** (10-20): Quick results, may underfit
- **Many epochs** (100-150): Better learning, takes longer

### Experiment 3: Sample Size
- **Small samples** (50): Quick generation, limited diversity
- **Large samples** (300+): More candidates, better exploration

## Troubleshooting

### Common Issues

**"Module not found" errors:**
- Click "Runtime" → "Restart runtime"
- Re-run the import cell

**Slow training:**
- Reduce epochs using the slider
- Use GPU runtime (Runtime → Change runtime type → GPU)

**Memory errors:**
- Reduce batch size in training cell (change 32 to 16)
- Generate fewer samples

**No output from cells:**
- Make sure cells are executed in order
- Check for error messages in red

### Colab-Specific Tips

- **Save frequently**: Use File → Save a copy in Drive
- **GPU acceleration**: Runtime → Change runtime type → GPU
- **Share results**: File → Download → Download .ipynb

## Expected Results

After successful completion, you should have:
- ✅ Trained VAE model
- ✅ Generated novel alloy compositions
- ✅ Visual comparisons of materials
- ✅ Clustering analysis
- ✅ Exported CSV of candidates

## Discussion Questions

1. How does changing the latent dimension affect material diversity?
2. What patterns do you see in the clustering results?
3. How realistic do the generated properties seem?
4. What additional material properties would you include?

## Next Steps

- **Validate experimentally**: Test promising candidates in lab
- **Extend to ternaries**: Modify code for 3-element alloys
- **Add constraints**: Incorporate physical feasibility rules
- **Optimize properties**: Use reinforcement learning for targeted design

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Restart runtime and try again
3. Review error messages carefully
4. Ask workshop facilitators for help

**Happy materials discovery! 🔬🤖**
