# TradingStrategist

> A modular machine learning framework for algorithmic trading strategy development and backtesting

## 📋 Overview

This repository contains **TradingStrategist**, a comprehensive machine learning-based trading strategy platform that implements and evaluates multiple algorithmic trading approaches. The framework allows for:

- Developing and testing manual trading strategies with technical indicators
- Training decision tree-based trading strategy models
- Implementing Q-learning reinforcement learning for trading
- Backtesting strategies with realistic market conditions including impact and commissions
- Comparing performance across different approach types

## 🚀 Key Features

| Category | What's Included |
|----------|-----------------|
| **Modular Package** | Organized structure with models, indicators, data handling, and simulation components |
| **Config-Driven Training** | YAML configuration files for reproducible experiments and consistent parameter management |
| **Multiple Strategy Types** | Manual rules-based, decision tree learning, and Q-learning reinforcement learning approaches |
| **Market Simulation** | Realistic market simulation with transaction costs, slippage, and commissions |
| **Technical Indicators** | Extensive collection of technical indicators (RSI, Bollinger Bands, MACD, etc.) |
| **Performance Analysis** | Sharpe ratio, cumulative returns, and other financial metrics |
| **Visualization Tools** | Plot comparison between strategies, benchmark performance, and trade timing |

## 🗂️ Repository Layout

```
├── configs/               # YAML configuration files for strategies and experiments
├── src/
│   └── TradingStrategist/
│       ├── __init__.py
│       ├── data/          # Data loading and preprocessing
│       ├── models/        # Strategy models (Manual, TreeStrategy, QStrategy)
│       ├── indicators/    # Technical indicators implementation
│       ├── simulation/    # Market simulation engine
│       ├── experiments/   # Experiment scripts for comparing strategies
│       ├── utils/         # Helper utilities 
│       ├── train.py       # Training entry point
│       └── evaluate.py    # Evaluation entry point
├── tests/                 # Unit tests
├── output/                # Generated model files and results
│   ├── models/            # Saved trained models
│   └── figures/           # Performance visualization outputs
├── Makefile               # Automation of common tasks
├── requirements.txt
└── README.md              # You are here
```

## ⚙️ Installation

```bash
# Clone the repository
$ git clone https://github.com/your-handle/tradingstrategist.git
$ cd tradingstrategist

# Create a virtual environment (using venv or conda)
$ python -m venv tradingstrategist-env
$ source tradingstrategist-env/bin/activate  # On Windows use tradingstrategist-env\Scripts\activate

# Install dependencies
$ pip install -r requirements.txt
$ pip install -e .  # Install package in development mode
```

## 📈 Quick Start

### Using the Makefile

```bash
# Run experiment comparing manual strategy with TreeStrategyLearner
$ make experiment1

# Run experiment testing impact of transaction costs
$ make experiment2

# Train a Q-learning strategy model
$ make train-q
```

### Using Python Modules

```bash
# Train a TreeStrategyLearner model
$ python -m TradingStrategist.train --config configs/tree_strategy.yaml

# Evaluate a trained model
$ python -m TradingStrategist.evaluate --config configs/tree_strategy.yaml

# Run manual strategy evaluation
$ python -m TradingStrategist.experiments.manual_strategy_evaluation --config configs/manual_strategy_config.yaml
```

## 🧪 Experiments

The package includes several experiment scripts:

1. **Manual vs. Tree Strategy Learner**: Compare performance of hand-crafted rules vs. machine learning
   ```bash
   $ make experiment1
   ```

2. **Impact of Transaction Costs**: Test how different market impact values affect strategy performance
   ```bash
   $ make experiment2
   ```

3. **Manual Strategy Evaluation**: Detailed analysis of technical indicator-based strategy
   ```bash
   $ make manual-strategy
   ```

## 🔧 Configuration

Strategies are configured through YAML files in the `configs` directory:

```yaml
# Example configuration
experiment:
  name: "TreeStrategy Evaluation"
  output_prefix: "tree_strategy"

data:
  symbol: "JPM"
  in_sample:
    start_date: "2008-01-01"
    end_date: "2009-12-31"
  out_sample:
    start_date: "2010-01-01"
    end_date: "2010-12-31"

dt_strategy:
  leaf_size: 5
  bags: 20
  window_size: 20
  prediction_days: 5
```

## 📊 Visualization

Performance visualization is automatically generated, showing:
- Normalized portfolio values over time
- Trade entry/exit points
- Comparison against buy-and-hold benchmark

Example output is saved to `output/figures/`.

## 🧑‍💻 Contributing

1. Fork the repository and create your feature branch
2. Make your changes and add tests if applicable
3. Run `make lint` and `make test` to ensure code quality
4. Submit a pull request with clear documentation of changes

## 📄 License

This project is licensed under the MIT License – see the LICENSE file for details.

## 🎓 Academic Integrity Notice

While derived from academic coursework, all refactorings, extensions, and additional material in this repository are original work and are provided **solely for portfolio demonstration**. Original assignment content has been substantially rewritten to comply with academic integrity policies.

## 🙏 Acknowledgements

- Original inspiration from Machine Learning for Trading course at Georgia Tech
- Open source libraries: NumPy, Pandas, SciPy, Matplotlib, scikit-learn