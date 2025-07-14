@echo off
REM Create virtual environment
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

REM Create output directories
mkdir output\images
mkdir output\reports
mkdir output\models

echo Setup complete! Activate virtual environment with: venv\Scripts\activate.bat
