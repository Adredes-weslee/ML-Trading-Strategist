"""
Training Script for TradingStrategist

This script provides an interface to train different trading strategy models
and save the trained models for later use, using YAML configuration files.
"""

import yaml
import argparse
import datetime as dt
import os
import pickle
from pathlib import Path

from TradingStrategist.models.TreeStrategyLearner import TreeStrategyLearner
from TradingStrategist.models.QStrategyLearner import QStrategyLearner
from TradingStrategist.utils.helpers import get_output_path, load_config


def train_tree_strategy_learner(symbol, start_date, end_date, impact, verbose, leaf_size=5, bags=20, 
                          window_size=20, rsi_window=14, stoch_window=14, cci_window=20, 
                          buy_threshold=0.02, sell_threshold=-0.02, prediction_days=5, 
                          position_size=1000, **kwargs):
    """
    Train a TreeStrategyLearner model and save it.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol to train on
    start_date : datetime
        Training start date
    end_date : datetime
        Training end date
    impact : float
        Market impact to consider
    verbose : bool
        Whether to print detailed output
    leaf_size : int, optional
        Leaf size for decision trees, default 5
    bags : int, optional
        Number of bags for ensemble, default 20
    window_size : int, optional
        Window size for indicators, default 20
    rsi_window : int, optional
        Window size for RSI, default 14
    stoch_window : int, optional
        Window size for stochastic oscillator, default 14
    cci_window : int, optional
        Window size for CCI, default 20
    buy_threshold : float, optional
        Threshold for buy signals, default 0.02
    sell_threshold : float, optional
        Threshold for sell signals, default -0.02
    prediction_days : int, optional
        Days to look ahead for return calculation, default 5
    position_size : int, optional
        Number of shares per trade, default 1000
    **kwargs : dict
        Additional parameters to pass to TreeStrategyLearner
        
    Returns:
    --------
    str
        Path to saved model file
    """
    print(f"Training TreeStrategyLearner on {symbol} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Create and train the learner with all parameters directly in constructor
    learner = TreeStrategyLearner(
        verbose=verbose, 
        impact=impact,
        leaf_size=leaf_size,
        bags=bags,
        window_size=window_size,
        rsi_window=rsi_window,
        stoch_window=stoch_window,
        cci_window=cci_window,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        prediction_days=prediction_days,
        position_size=position_size,
        **kwargs  # Pass any additional parameters
    )
    
    # Train the model
    learner.addEvidence(symbol=symbol, sd=start_date, ed=end_date)
    
    # Save the trained model
    output_dir = get_output_path() / "models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = output_dir / f"tree_strategy_learner_{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(learner, f)
    
    print(f"Model saved to: {model_path}")
    return str(model_path)


def train_q_strategy_learner(symbol, start_date, end_date, impact, verbose, 
                           learning_rate=0.2, discount_factor=0.9, 
                           random_action_rate=0.5, random_action_decay=0.99,
                           dyna_iterations=10, **kwargs):
    """
    Train a QStrategyLearner model and save it.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol to train on
    start_date : datetime
        Training start date
    end_date : datetime
        Training end date
    impact : float
        Market impact to consider
    verbose : bool
        Whether to print detailed output
    learning_rate : float, optional
        Q-learning alpha parameter, default 0.2
    discount_factor : float, optional
        Q-learning gamma parameter, default 0.9
    random_action_rate : float, optional
        Initial exploration rate, default 0.5
    random_action_decay : float, optional
        Rate at which exploration decays, default 0.99
    dyna_iterations : int, optional
        Number of Dyna-Q planning iterations, default 10
    **kwargs : dict
        Additional parameters to pass to QStrategyLearner
        
    Returns:
    --------
    str
        Path to saved model file
    """
    print(f"Training QStrategyLearner on {symbol} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Create and train the learner with all parameters directly in constructor
    learner = QStrategyLearner(
        verbose=verbose, 
        impact=impact,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        random_action_rate=random_action_rate,
        random_action_decay=random_action_decay,
        dyna_iterations=dyna_iterations,
        **kwargs  # Pass any additional parameters
    )
    
    # Train the model
    learner.addEvidence(symbol=symbol, sd=start_date, ed=end_date)
    
    # Save the trained model
    output_dir = get_output_path() / "models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = output_dir / f"q_strategy_learner_{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(learner, f)
    
    print(f"Model saved to: {model_path}")
    return str(model_path)


def train_tree_strategy_learner_from_config(config_path):
    """
    Train a TreeStrategyLearner using parameters from a config file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
        
    Returns:
    --------
    str
        Path to saved model file
    """
    config = load_config(config_path)
    
    # Extract parameters from config
    symbol = config['data']['symbol']
    
    # Get training dates - handle all possible date formats
    if 'training' in config['data']:
        start_date = dt.datetime.strptime(config['data']['training']['start_date'], '%Y-%m-%d')
        end_date = dt.datetime.strptime(config['data']['training']['end_date'], '%Y-%m-%d')
    elif 'in_sample' in config['data']:
        start_date = dt.datetime.strptime(config['data']['in_sample']['start_date'], '%Y-%m-%d')
        end_date = dt.datetime.strptime(config['data']['in_sample']['end_date'], '%Y-%m-%d')
    else:
        # Direct date format
        start_date = dt.datetime.strptime(config['data']['start_date'], '%Y-%m-%d')
        end_date = dt.datetime.strptime(config['data']['end_date'], '%Y-%m-%d')
    
    # Get trading parameters
    trading_config = config.get('trading', {})
    impact = trading_config.get('impact', 0.005)
    
    # Get strategy learner parameters - check both naming conventions
    if 'tree_strategy_learner' in config:
        sl_params = config['tree_strategy_learner']
    elif 'dt_strategy' in config:
        sl_params = config['dt_strategy']
    elif 'strategy_learner' in config:
        sl_params = config['strategy_learner']
    else:
        sl_params = {}
    
    # Extract specific parameters with defaults
    params = {
        'leaf_size': sl_params.get('leaf_size', 5),
        'bags': sl_params.get('bags', 20),
        'window_size': sl_params.get('window_size', 20),
        'rsi_window': sl_params.get('rsi_window', 14),
        'stoch_window': sl_params.get('stoch_window', 14),
        'cci_window': sl_params.get('cci_window', 20),
        'buy_threshold': sl_params.get('buy_threshold', 0.02),
        'sell_threshold': sl_params.get('sell_threshold', -0.02),
        'prediction_days': sl_params.get('prediction_days', 5),
        'position_size': sl_params.get('position_size', 1000),
        'verbose': config.get('verbose', False)
    }
    
    # Add any other parameters from sl_params
    for key, value in sl_params.items():
        if key not in params:
            params[key] = value
    
    # Train the model with extracted parameters
    return train_tree_strategy_learner(
        symbol=symbol, 
        start_date=start_date, 
        end_date=end_date, 
        impact=impact, 
        **params  # Pass all parameters unpacked
    )


def train_q_strategy_learner_from_config(config_path):
    """
    Train a QStrategyLearner using parameters from a config file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
        
    Returns:
    --------
    str
        Path to saved model file
    """
    config = load_config(config_path)
    
    # Extract parameters from config
    symbol = config['data']['symbol']
    
    # Get training dates - handle all possible date formats
    if 'training' in config['data']:
        start_date = dt.datetime.strptime(config['data']['training']['start_date'], '%Y-%m-%d')
        end_date = dt.datetime.strptime(config['data']['training']['end_date'], '%Y-%m-%d')
    elif 'in_sample' in config['data']:
        start_date = dt.datetime.strptime(config['data']['in_sample']['start_date'], '%Y-%m-%d')
        end_date = dt.datetime.strptime(config['data']['in_sample']['end_date'], '%Y-%m-%d')
    else:
        # Direct date format
        start_date = dt.datetime.strptime(config['data']['start_date'], '%Y-%m-%d')
        end_date = dt.datetime.strptime(config['data']['end_date'], '%Y-%m-%d')
    
    # Get trading parameters
    trading_config = config.get('trading', {})
    impact = trading_config.get('impact', 0.005)
    
    # Get Q-strategy parameters
    q_config = config.get('q_strategy', {})
    
    # Extract all parameters with defaults
    params = {
        'learning_rate': q_config.get('learning_rate', 0.2),
        'discount_factor': q_config.get('discount_factor', 0.9),
        'random_action_rate': q_config.get('random_action_rate', 0.5),
        'random_action_decay': q_config.get('random_action_decay', 0.99),
        'dyna_iterations': q_config.get('dyna_iterations', 10),
        'verbose': config.get('verbose', False)
    }
    
    # Add any other parameters from q_config
    for key, value in q_config.items():
        if key not in params:
            params[key] = value
    
    # Train the model with extracted parameters
    return train_q_strategy_learner(
        symbol=symbol, 
        start_date=start_date, 
        end_date=end_date, 
        impact=impact, 
        **params  # Pass all parameters unpacked
    )


def train_multiple_impact_models(config_path):
    """
    Train multiple TreeStrategyLearner models with different impact values.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
        
    Returns:
    --------
    list
        List of paths to saved model files
    """
    config = load_config(config_path)
    
    # Extract parameters from config
    symbol = config['data']['symbol']
    
    # Get training dates - handle all possible date formats
    if 'training' in config['data']:
        start_date = dt.datetime.strptime(config['data']['training']['start_date'], '%Y-%m-%d')
        end_date = dt.datetime.strptime(config['data']['training']['end_date'], '%Y-%m-%d')
    elif 'in_sample' in config['data']:
        start_date = dt.datetime.strptime(config['data']['in_sample']['start_date'], '%Y-%m-%d')
        end_date = dt.datetime.strptime(config['data']['in_sample']['end_date'], '%Y-%m-%d')
    else:
        # Direct date format
        start_date = dt.datetime.strptime(config['data']['start_date'], '%Y-%m-%d')
        end_date = dt.datetime.strptime(config['data']['end_date'], '%Y-%m-%d')
    
    # Get strategy learner parameters - check both naming conventions
    if 'tree_strategy_learner' in config:
        sl_params = config['tree_strategy_learner']
    elif 'dt_strategy' in config:
        sl_params = config['dt_strategy']
    elif 'strategy_learner' in config:
        sl_params = config['strategy_learner']
    else:
        sl_params = {}
    
    # Extract all parameters with defaults
    params = {
        'leaf_size': sl_params.get('leaf_size', 5),
        'bags': sl_params.get('bags', 20),
        'window_size': sl_params.get('window_size', 20),
        'rsi_window': sl_params.get('rsi_window', 14),
        'stoch_window': sl_params.get('stoch_window', 14),
        'cci_window': sl_params.get('cci_window', 20),
        'buy_threshold': sl_params.get('buy_threshold', 0.02),
        'sell_threshold': sl_params.get('sell_threshold', -0.02),
        'prediction_days': sl_params.get('prediction_days', 5),
        'position_size': sl_params.get('position_size', 1000),
        'verbose': config.get('verbose', False)
    }
    
    # Add any other parameters from sl_params
    for key, value in sl_params.items():
        if key not in params and key not in ['impact', 'commission']:
            params[key] = value
    
    # Get impact values to test - handle both naming conventions
    trading_config = config.get('trading', {})
    impact_values = (
        trading_config.get('impacts') or 
        trading_config.get('impact_values') or 
        [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
    )
    
    # Train a model for each impact value
    model_paths = []
    for impact in impact_values:
        print(f"\nTraining TreeStrategyLearner with impact={impact}")
        model_path = train_tree_strategy_learner(
            symbol=symbol, 
            start_date=start_date, 
            end_date=end_date, 
            impact=impact, 
            **params  # Pass all parameters unpacked
        )
        model_paths.append(model_path)
    
    return model_paths


def main():
    """Main function to load config file and train appropriate model(s)."""
    parser = argparse.ArgumentParser(description='Train trading strategy models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    config_path = args.config
    print(f"Loading configuration from {config_path}")
    
    # Choose appropriate training function based on config filename
    if 'experiment2' in config_path:
        # For experiment 2, train multiple models with different impacts
        train_multiple_impact_models(config_path)
    elif 'qstrategy' in config_path:
        # For Q-strategy config
        train_q_strategy_learner_from_config(config_path)
    else:
        # Default to regular strategy learner
        train_tree_strategy_learner_from_config(config_path)


if __name__ == "__main__":
    main()