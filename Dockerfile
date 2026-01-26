# Use Python 3.12 slim image for numpy 2.4.1 compatibility
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for faster builds and to avoid lxml compilation issues
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api_authentication_handler.py .
COPY centralized_field_mapping.py .
COPY data_generator.py .
COPY debug_field_mapping.py .
COPY export_for_lab.py .
COPY feature_schema.yml .
COPY field_mapping_utils.py .
COPY gradio_app.py .
COPY hazards.yml .
COPY materials_discovery_model.py .
COPY materials_discovery_api.py .
COPY synthesizability_predictor.py .
COPY test_docker_solution.py .
COPY test_field_mapping_fixes.py .
COPY test_ml_prediction.py .
COPY test_mp_end_to_end.py .
COPY test_mp_integration.py .
COPY test_synthesizability.py .

# Expose port 8080 for Google Cloud Run
EXPOSE 8080

# Set minimum memory limit to 1Gi for Google Cloud Run
ENV CLOUD_RUN_MEMORY_LIMIT=1Gi

# Enable Docker debug logging and unbuffered Python output
ENV DOCKER_DEBUG=1
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "gradio_app.py"]
