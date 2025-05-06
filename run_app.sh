#!/bin/bash
echo "Installing required packages..."
python -m pip install --upgrade pip
python -m pip install setuptools wheel
python -m pip install -r requirements.txt
echo ""
echo "Installing the package in development mode..."
python -m pip install -e .
echo ""
echo "Starting TradingStrategist simplified application..."
streamlit run simple_app.py