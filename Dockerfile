# Use Python 3.10 slim image
FROM python:3.10-slim

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
COPY gradio_app.py .
COPY synthesizability_predictor.py .
COPY export_for_lab.py .

# Expose port 8080 for Google Cloud Run
EXPOSE 8080

# Run the application
CMD ["python", "gradio_app.py"]
