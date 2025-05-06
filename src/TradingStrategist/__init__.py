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
__author__ = "Wes"

# Core components - using relative imports consistently
from .models.ManualStrategy import ManualStrategy
from .models.TreeStrategyLearner import TreeStrategyLearner
from .models.QStrategyLearner import QStrategyLearner
from .indicators.technical import Indicators
from .simulation.market_sim import compute_portvals, assess_portfolio, compute_portfolio_stats
from .data.loader import get_data
from .utils.helpers import load_config, get_output_path

# Make key components available at the package level
__all__ = [
    'ManualStrategy',
    'TreeStrategyLearner',
    'QStrategyLearner',
    'Indicators',
    'compute_portvals',
    'assess_portfolio',
    'compute_portfolio_stats',
    'get_data',
    'load_config',
    'get_output_path',
]