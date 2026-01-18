#!/bin/bash

echo "========================================"
echo "Materials Discovery ML Workshop - Linux"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed."
    echo "Please install Python 3.8+ using your package manager:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "  Fedora: sudo dnf install python3 python3-pip"
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to create virtual environment."
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment."
    exit 1
fi

# Install/update requirements
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies."
    exit 1
fi

echo
echo "========================================"
echo "Installation complete!"
echo "========================================"
echo
echo "Choose your deployment option:"
echo "1. Jupyter Notebook Workshop (Educational)"
echo "2. Streamlit Web App (Production)"
echo

read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo
    echo "Launching Jupyter Notebook Workshop..."
    echo "Open the materials_discovery_workshop.ipynb file when Jupyter opens."
    jupyter notebook materials_discovery_workshop.ipynb
elif [ "$choice" = "2" ]; then
    echo
    echo "Launching Streamlit Web App..."
    echo "Open http://localhost:8501 in your browser."
    streamlit run app.py
else
    echo "Invalid choice. Exiting."
    exit 1
fi
