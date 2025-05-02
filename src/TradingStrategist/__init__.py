"""
TradingStrategist - A modular implementation of machine learning-based trading strategies

This package provides tools for:
- Technical indicator calculation and analysis
- Machine learning models for trading decisions
- Strategy evaluation and comparison
- Market simulation with transaction costs
"""

# Version information
__version__ = "0.1.0"
__author__ = "Wes"  # Replace with your name

# Core components
# Fixed imports to match actual file names/class names
from TradingStrategist.models.ManualStrategy import ManualStrategy  # Fixed CamelCase
from TradingStrategist.models.TreeStrategyLearner import TreeStrategyLearner  # Fixed class name
from TradingStrategist.models.QStrategyLearner import QStrategyLearner  # Assuming this one is correct
from TradingStrategist.indicators.technical import Indicators
from TradingStrategist.simulation.market_sim import compute_portvals, assess_portfolio

# Make key components available at the package level
__all__ = [
    'ManualStrategy',
    'TreeStrategyLearner',  # Updated to match the actual class name
    'QStrategyLearner',
    'Indicators',
    'compute_portvals',
    'assess_portfolio',
]