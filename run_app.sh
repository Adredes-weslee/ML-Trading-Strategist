#!/bin/bash
echo "Installing required packages..."
python -m pip install -r requirements.txt
echo ""
echo "Starting TradingStrategist application..."
streamlit run app.py