"""
Experiment scripts for strategy comparison and analysis.
"""

# Export key functions for external use
from .manual_strategy_evaluation import run_manual_strategy_test, run_evaluation_from_config

__all__ = [
    'run_manual_strategy_test',
    'run_evaluation_from_config'  # Updated function name
]