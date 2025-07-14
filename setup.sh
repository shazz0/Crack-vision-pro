#!/bin/bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create output directories
mkdir -p output/images
mkdir -p output/reports
mkdir -p output/models

echo "Setup complete! Activate virtual environment with: source venv/bin/activate"
