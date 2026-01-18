# Materials Discovery Workshop - Gradio Web App Deployment Guide

This guide provides instructions for deploying the Materials Discovery Workshop Gradio web application to Google Cloud Run and HuggingFace Spaces.

## 🚀 Application Overview

The Gradio app provides an interactive interface for:

- **VAE-based material generation** using machine learning
- **Synthesizability prediction** with ML and rule-based models
- **Priority ranking** for experimental synthesis
- **Cost-benefit analysis** for synthesis planning
- **Experimental workflow** generation

## 📋 Prerequisites

- Google Cloud SDK (for Cloud Run deployment)
- Docker (for containerization)
- HuggingFace account (for Spaces deployment)
- Python 3.9+ with required dependencies

## 🐳 Google Cloud Run Deployment

### 1. Build and Push Docker Image

```bash
# Build the Docker image
docker build -t materials-discovery-workshop .

# Tag for Google Container Registry
docker tag materials-discovery-workshop gcr.io/YOUR_PROJECT_ID/materials-discovery-workshop

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/materials-discovery-workshop
```

### 2. Deploy to Cloud Run

```bash
# Deploy the container
gcloud run deploy materials-discovery-workshop \
  --image gcr.io/YOUR_PROJECT_ID/materials-discovery-workshop \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 7860 \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10 \
  --timeout 3600
```

### 3. Access Your Application

After deployment, Cloud Run will provide a URL where your app is accessible.

## 🤗 HuggingFace Spaces Deployment

### Option 1: Direct Upload via Web Interface

1. Go to [HuggingFace Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **Space name**: `materials-discovery-workshop`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU (or GPU for faster processing)
4. Upload the following files:
   - `app.py`
   - `gradio_app.py`
   - `requirements.txt`

### Option 2: Using Git (Recommended)

1. Create a new Space on HuggingFace
2. Clone the repository:

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/materials-discovery-workshop
cd materials-discovery-workshop
```

1. Copy the application files:

```bash
cp /path/to/your/local/gradio_app.py .
cp /path/to/your/local/app.py .
cp /path/to/your/local/requirements.txt .
```

1. Commit and push:

```bash
git add .
git commit -m "Initial deployment of Materials Discovery Workshop"
git push
```

## 🔧 Configuration Options

### Environment Variables

For production deployments, you can set:

```bash
# For Cloud Run
gcloud run deploy ... --set-env-vars GRADIO_SERVER_NAME=0.0.0.0,GRADIO_SERVER_PORT=7860
```

### Scaling

- **Cloud Run**: Automatically scales based on traffic
- **Spaces**: Limited by HuggingFace's free tier (upgrade for more resources)

## 🚀 Local Development and Testing

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python gradio_app.py
```

### Test with Docker

```bash
# Build and run locally
docker build -t materials-discovery-workshop .
docker run -p 7860:7860 materials-discovery-workshop
```

## 📊 Performance Considerations

### Memory Usage

- **Base memory**: ~2GB RAM
- **Per user session**: ~500MB additional
- **Model loading**: ~1GB for ML models

### CPU/GPU Requirements

- **CPU**: 2+ cores recommended
- **GPU**: Optional, speeds up VAE training (CUDA support included)

## 🔒 Security Considerations

### Cloud Run

- Uses Google Cloud's built-in security
- HTTPS enabled by default
- IAM permissions for access control

### HuggingFace Spaces

- Public access by default
- Rate limiting applied
- Content moderation enabled

## 📈 Monitoring and Maintenance

### Cloud Run Monitoring

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=materials-discovery-workshop"

# View metrics
gcloud monitoring dashboards create --config-from-file=dashboard.json
```

### Spaces Monitoring

- Access logs via HuggingFace dashboard
- Monitor usage and performance metrics

## 🛠️ Troubleshooting

### Common Issues

1. **Memory errors**: Increase memory allocation in Cloud Run
2. **Timeout errors**: Increase timeout settings for long-running generations
3. **Import errors**: Ensure all dependencies are in requirements.txt
4. **CUDA errors**: Application falls back to CPU if GPU unavailable

### Debug Mode

Enable debug logging by modifying the app launch:

```python
interface.launch(debug=True)
```

## 📚 File Structure

```path
materials-discovery-workshop/
├── gradio_app.py          # Main application code
├── app.py                 # HuggingFace Spaces entry point
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
└── GRADIO_DEPLOYMENT_README.md  # This file
```

## 🎯 Next Steps

1. **Extend with real data**: Integrate Materials Project API for real materials
2. **Add authentication**: Implement user management and access control
3. **Database integration**: Store results and user sessions
4. **API endpoints**: Add REST API for programmatic access
5. **Advanced ML models**: Implement more sophisticated generative models

## 📞 Support

For issues and questions:

- Check the [GitHub repository](https://github.com/jmeyer1980/materials-discovery-workshop)
- Open issues for bugs and feature requests
- Review deployment logs for runtime errors

---

- **Happy deploying! 🚀🧪**
