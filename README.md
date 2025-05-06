# TradingStrategist 📈

> A modular machine learning framework for algorithmic trading strategy development and backtesting

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ML-Trading-Strategist.git
cd ML-Trading-Strategist

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

#### On Windows (PowerShell):
```powershell
./run_streamlit.ps1
```

#### On Windows (Command Prompt):
```cmd
run_app.bat
```

#### On Linux/macOS:
```bash
# Make the script executable (first time only)
chmod +x run_app.sh

# Run the application
./run_app.sh
```

The application will open in your default web browser at http://localhost:8501

## 📋 Project Overview

TradingStrategist is a comprehensive machine learning-based trading strategy platform that implements and evaluates multiple algorithmic trading approaches:

- Manual rule-based strategies using technical indicators
- Decision tree-based machine learning strategies
- Q-learning reinforcement learning for trading
- Backtesting with realistic market conditions including impact and commissions

## 🖥️ Using the Interface

The Streamlit interface allows you to:

1. **Select Stock Symbols** - Choose from various stock data included in the repository
2. **Set Date Ranges** - Configure training and testing periods
3. **Select Strategies** - Compare different trading approaches:
   - Benchmark (buy and hold)
   - Manual Strategy (rule-based)
   - Tree Strategy Learner (decision tree ensemble)
   - Q-Strategy Learner (reinforcement learning)
4. **Tune Parameters** - Customize each strategy's parameters
5. **Analyze Results** - View performance metrics and visualizations

## 🔍 Key Features

| Category | What's Included |
|----------|-----------------|
| **Multiple Strategy Types** | Manual rules-based, decision tree learning, and Q-learning approaches |
| **Technical Indicators** | RSI, Bollinger Bands, MACD, Stochastic Oscillator, and more |
| **Market Simulation** | Transaction costs, slippage, and commissions |
| **Performance Metrics** | Sharpe ratio, cumulative returns, and other financial metrics |
| **Interactive Interface** | Streamlit web application for easy experimentation |

## 🧩 Project Structure

```
├── app.py                 # Streamlit application
├── requirements.txt       # Dependencies
├── run_streamlit.ps1      # Windows PowerShell launch script
├── run_app.bat            # Windows Command Prompt launch script
├── run_app.sh             # Linux/macOS launch script
├── configs/               # YAML configuration files
├── data/                  # Stock price data CSV files
└── src/                   # TradingStrategist implementation
    └── TradingStrategist/
        ├── data/          # Data loading and preprocessing
        ├── models/        # Strategy models (Manual, TreeStrategy, QStrategy)
        ├── indicators/    # Technical indicators implementation
        ├── simulation/    # Market simulation engine
        ├── experiments/   # Experiment scripts for comparing strategies
        └── utils/         # Helper utilities
```

## 🔬 Advanced Usage

For advanced users who want to extend the framework beyond the Streamlit interface:

```python
# Example: Using the TreeStrategyLearner programmatically
from src.TradingStrategist.models.TreeStrategyLearner import TreeStrategyLearner
from datetime import datetime

# Create and train a model
learner = TreeStrategyLearner(leaf_size=5, bags=20)
learner.addEvidence(
    symbol="AAPL",
    sd=datetime(2008, 1, 1),
    ed=datetime(2009, 12, 31)
)

# Test the model
trades = learner.testPolicy(
    symbol="AAPL",
    sd=datetime(2010, 1, 1),
    ed=datetime(2010, 12, 31)
)
```

## 📊 Model Types

### Manual Strategy
A rules-based approach using technical indicators like RSI, Bollinger Bands, and MACD to generate trading signals based on predefined thresholds.

### Tree Strategy Learner
A machine learning approach using Random Decision Trees with bagging to create an ensemble that predicts future price movements and generates trading signals.

### Q-Strategy Learner
A reinforcement learning approach where a Q-learning algorithm learns optimal trading actions by interacting with a simulated market environment.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Built with: NumPy, Pandas, scikit-learn, Streamlit, Matplotlib