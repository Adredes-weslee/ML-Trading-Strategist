"""
TradingStrategist Web Application

This Streamlit application provides a user interface for working with the
TradingStrategist machine learning framework for algorithmic trading strategy
development and backtesting.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yaml
from pathlib import Path

# Add the src directory to the path so we can import from the module structure
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Now import from the project with the standardized import pattern
from src.TradingStrategist.models.ManualStrategy import ManualStrategy
from src.TradingStrategist.models.TreeStrategyLearner import TreeStrategyLearner
from src.TradingStrategist.models.QStrategyLearner import QStrategyLearner
from src.TradingStrategist.data.loader import get_data
from src.TradingStrategist.simulation.market_sim import compute_portvals, compute_portfolio_stats
from src.TradingStrategist.utils.helpers import load_config

# Configure the page
st.set_page_config(
    page_title="TradingStrategist",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Function to load configuration files
def load_yaml_config(config_file):
    """
    Load configuration from a YAML file.
    
    Parameters:
    -----------
    config_file : str
        Path to the configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        st.error(f"Error loading configuration file {config_file}: {str(e)}")
        return {}

# Function to save configuration to YAML file
def save_yaml_config(config, config_file):
    """
    Save configuration to a YAML file.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    config_file : str
        Path to save the configuration file
    """
    try:
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        st.error(f"Error saving configuration file {config_file}: {str(e)}")
        return False

# Load default configuration files
@st.cache_data
def load_default_configs():
    """
    Load all default configurations.
    
    Returns:
    --------
    dict
        Dictionary containing all configurations
    """
    config_dir = os.path.join(project_root, 'configs')
    
    configs = {
        'manual': load_yaml_config(os.path.join(config_dir, 'manual_strategy_config.yaml')),
        'tree': load_yaml_config(os.path.join(config_dir, 'tree_strategy.yaml')),
        'qlearn': load_yaml_config(os.path.join(config_dir, 'qstrategy.yaml')),
        'data': load_yaml_config(os.path.join(config_dir, 'data.yaml')),
        'market_sim': load_yaml_config(os.path.join(config_dir, 'market_sim.yaml'))
    }
    
    return configs

# Title and description
st.title("📈 TradingStrategist")
st.markdown(
    """
    A modular machine learning framework for algorithmic trading strategy development and backtesting.
    
    This application allows you to:
    
    - Develop and test manual trading strategies with technical indicators
    - Train decision tree-based trading strategy models
    - Implement Q-learning reinforcement learning for trading
    - Backtest strategies with realistic market conditions including impact and commissions
    - Compare performance across different approach types
    """
)

# Load configurations
configs = load_default_configs()

@st.cache_data
def get_available_symbols():
    """Get the list of available stock symbols in the data directory."""
    data_dir = Path("data")
    if not data_dir.exists():
        return []
    
    symbols = []
    for file_path in data_dir.glob("*.csv"):
        if file_path.name not in ["$DJI.csv", "$SPX.csv", "$VIX.csv"]:  # Skip indices
            symbols.append(file_path.stem)
    return sorted(symbols)

@st.cache_data
def calculate_metrics(portfolio_values):
    """
    Calculate performance metrics for a portfolio.
    
    Parameters:
    -----------
    portfolio_values : pd.DataFrame
        DataFrame containing portfolio values
        
    Returns:
    --------
    dict
        Dictionary containing performance metrics
    """
    # Calculate daily returns
    daily_returns = portfolio_values.pct_change().dropna()
    
    # Calculate metrics
    metrics = {
        'cumulative_return': float((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1),
        'average_daily_return': float(daily_returns.mean()),
        'std_daily_return': float(daily_returns.std()),
        'sharpe_ratio': float(np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std().iloc[0] > 0 else 0),
        'final_value': float(portfolio_values.iloc[-1].iloc[0])
    }
    
    return metrics

def create_benchmark(symbol, start_date, end_date, starting_value, commission, impact):
    """
    Create a benchmark portfolio (buy and hold strategy).
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    start_date : datetime
        Start date
    end_date : datetime
        End date
    starting_value : int
        Starting portfolio value
    commission : float
        Commission per trade
    impact : float
        Market impact per trade
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing benchmark portfolio values
    """
    # Get price data
    dates = pd.date_range(start_date, end_date)
    prices = get_data([symbol], dates)
    
    # Create trades DataFrame for benchmark
    trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
    trades.iloc[0] = 1000  # Buy 1000 shares on first day and hold
    
    # Calculate portfolio values
    benchmark_values = compute_portvals(orders=trades, 
                                       start_val=starting_value, 
                                       commission=commission, 
                                       impact=impact)
    
    return benchmark_values

def run_strategy(strategy_type, symbol, train_start_date, train_end_date, 
                 test_start_date, test_end_date, starting_value,
                 commission, impact, **strategy_params):
    """
    Run a trading strategy and return the portfolio values and metrics.
    
    Parameters:
    -----------
    strategy_type : str
        Type of strategy (manual, tree, qlearn)
    symbol : str
        Stock symbol
    train_start_date, train_end_date : datetime
        Training period dates
    test_start_date, test_end_date : datetime
        Testing period dates
    starting_value : float
        Initial portfolio value
    commission, impact : float
        Trading cost parameters
    **strategy_params : dict
        Additional parameters for specific strategies
        
    Returns:
    --------
    tuple
        (portfolio_values, trades, metrics)
    """
    if strategy_type == 'manual':
        model = ManualStrategy(
            verbose=False,
            **{k: v for k, v in strategy_params.items() if k in [
                'window_size', 'rsi_window', 'stoch_window', 'cci_window',
                'buy_threshold', 'sell_threshold', 'position_size'
            ]}
        )
        trades = model.testPolicy(
            symbol=symbol, 
            sd=test_start_date, 
            ed=test_end_date, 
            sv=starting_value
        )
    
    elif strategy_type == 'tree':
        model = TreeStrategyLearner(
            verbose=False,
            impact=impact,
            commission=commission,
            **{k: v for k, v in strategy_params.items() if k in [
                'window_size', 'buy_threshold', 'sell_threshold',
                'prediction_days', 'leaf_size', 'bags', 'position_size'
            ]}
        )
        
        # Train the model
        model.addEvidence(
            symbol=symbol,
            sd=train_start_date,
            ed=train_end_date,
            sv=starting_value
        )
        
        # Generate trades
        trades = model.testPolicy(
            symbol=symbol,
            sd=test_start_date,
            ed=test_end_date,
            sv=starting_value
        )
    
    elif strategy_type == 'qlearn':
        model = QStrategyLearner(
            verbose=False,
            impact=impact,
            commission=commission,
            **{k: v for k, v in strategy_params.items() if k in [
                'indicator_bins', 'window_size', 'rsi_window', 'position_size',
                'max_iterations', 'learning_rate', 'discount_factor',
                'random_action_rate', 'random_action_decay', 'dyna_iterations',
                'use_bb', 'use_rsi', 'use_macd', 'use_stoch', 'use_cci',
                'momentum_periods', 'convergence_threshold'
            ]}
        )
        
        # Train the model
        model.addEvidence(
            symbol=symbol,
            sd=train_start_date,
            ed=train_end_date,
            sv=starting_value
        )
        
        # Generate trades
        trades = model.testPolicy(
            symbol=symbol,
            sd=test_start_date,
            ed=test_end_date,
            sv=starting_value
        )
    
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    # Compute portfolio values
    portvals = compute_portvals(
        orders=trades,
        start_val=starting_value,
        commission=commission,
        impact=impact
    )
    
    # Calculate metrics
    metrics = calculate_metrics(portvals)
    
    return portvals, trades, metrics

# Sidebar for inputs
st.sidebar.header("Strategy Settings")

# Configuration management
with st.sidebar.expander("Configuration Management"):
    st.subheader("Configuration Options")
    
    # Load configuration option
    st.write("Use configuration files to set default parameters.")
    use_configs = st.checkbox("Use Configuration Files", value=True)
    
    # Save configuration option
    if st.button("Save Current Settings"):
        # Create configurations for saving
        manual_config = configs['manual'].copy() if 'manual' in configs else {}
        tree_config = configs['tree'].copy() if 'tree' in configs else {}
        qlearn_config = configs['qlearn'].copy() if 'qlearn' in configs else {}
        
        # Update manual strategy config if it exists
        if 'manual_strategy' in manual_config:
            manual_config['manual_strategy'].update({
                k: v for k, v in manual_params.items() if k in [
                    'window_size', 'rsi_window', 'buy_threshold', 
                    'sell_threshold', 'position_size'
                ]
            })
        
        # Update tree strategy config if it exists
        if 'tree_strategy_learner' in tree_config:
            tree_config['tree_strategy_learner'].update({
                k: v for k, v in tree_params.items() if k in [
                    'window_size', 'buy_threshold', 'sell_threshold',
                    'prediction_days', 'leaf_size', 'bags', 'position_size'
                ]
            })
        
        # Update Q-learning strategy config if it exists
        if 'q_strategy_learner' in qlearn_config:
            qlearn_config['q_strategy_learner'].update({
                k: v for k, v in q_params.items() if k in [
                    'indicator_bins', 'window_size', 'rsi_window', 'stoch_window',
                    'cci_window', 'position_size', 'max_iterations', 'learning_rate',
                    'discount_factor', 'random_action_rate', 'random_action_decay',
                    'dyna_iterations', 'convergence_threshold', 'use_bb', 'use_rsi',
                    'use_macd', 'use_stoch', 'use_cci', 'momentum_periods'
                ]
            })
        
        # Save configurations
        config_dir = os.path.join(project_root, 'configs')
        
        if 'manual_strategy' in manual_config:
            success_manual = save_yaml_config(
                manual_config, 
                os.path.join(config_dir, 'manual_strategy_config.yaml')
            )
            
        if 'tree_strategy_learner' in tree_config:
            success_tree = save_yaml_config(
                tree_config, 
                os.path.join(config_dir, 'tree_strategy.yaml')
            )
            
        if 'q_strategy_learner' in qlearn_config:
            success_qlearn = save_yaml_config(
                qlearn_config, 
                os.path.join(config_dir, 'qstrategy.yaml')
            )
            
        st.success("Settings saved successfully! Reload the page to use the new defaults.")

# Get available symbols
symbols = get_available_symbols()
if not symbols:
    st.error("No stock data found in the data directory.")
    st.stop()

# Get default symbol from config if available
default_symbol = "JPM"  # Default in case config doesn't specify
if use_configs and 'data' in configs and 'symbol' in configs['data']:
    default_symbol = configs['data']['symbol']

# Stock selection
selected_symbol = st.sidebar.selectbox("Select Stock Symbol", symbols, 
                                      index=symbols.index(default_symbol) if default_symbol in symbols else 0)

# Date range selection
st.sidebar.subheader("Date Range")
col1, col2 = st.sidebar.columns(2)

# Default dates
default_train_start = dt.datetime(2008, 1, 1)
default_train_end = dt.datetime(2009, 12, 31)
default_test_start = dt.datetime(2010, 1, 1)
default_test_end = dt.datetime(2011, 12, 31)

# Use dates from config if available and enabled
if use_configs and 'data' in configs:
    config_data = configs['data']
    if 'training' in config_data:
        if 'start_date' in config_data['training']:
            default_train_start = dt.datetime.strptime(config_data['training']['start_date'], '%Y-%m-%d')
        if 'end_date' in config_data['training']:
            default_train_end = dt.datetime.strptime(config_data['training']['end_date'], '%Y-%m-%d')
    if 'testing' in config_data:
        if 'start_date' in config_data['testing']:
            default_test_start = dt.datetime.strptime(config_data['testing']['start_date'], '%Y-%m-%d')
        if 'end_date' in config_data['testing']:
            default_test_end = dt.datetime.strptime(config_data['testing']['end_date'], '%Y-%m-%d')

min_date = dt.datetime(2007, 1, 1)
max_date = dt.datetime(2011, 12, 31)

with col1:
    train_start = st.date_input("Training Start Date", default_train_start, min_value=min_date, max_value=max_date)
    test_start = st.date_input("Testing Start Date", default_test_start, min_value=min_date, max_value=max_date)
with col2:
    train_end = st.date_input("Training End Date", default_train_end, min_value=min_date, max_value=max_date)
    test_end = st.date_input("Testing End Date", default_test_end, min_value=min_date, max_value=max_date)

# Portfolio settings
st.sidebar.subheader("Portfolio Settings")
default_starting_value = 100000
default_commission = 9.95
default_impact = 0.005

# Use portfolio settings from config if available
if use_configs:
    if 'portfolio' in configs.get('data', {}):
        if 'starting_value' in configs['data']['portfolio']:
            default_starting_value = configs['data']['portfolio']['starting_value']
    
    if 'trading' in configs.get('market_sim', {}):
        if 'commission' in configs['market_sim']['trading']:
            default_commission = configs['market_sim']['trading']['commission']
        if 'impact' in configs['market_sim']['trading']:
            default_impact = configs['market_sim']['trading']['impact']

starting_value = st.sidebar.number_input("Starting Portfolio Value", min_value=1000, max_value=1000000, value=default_starting_value, step=10000)
commission = st.sidebar.number_input("Commission per Trade ($)", min_value=0.0, max_value=50.0, value=default_commission, step=0.5)
impact = st.sidebar.number_input("Market Impact per Trade (%)", min_value=0.0, max_value=0.05, value=default_impact, step=0.001, format="%.3f")

# Strategy selection with standardized naming
selected_strategies = st.sidebar.multiselect(
    "Select Strategies to Compare",
    ["Benchmark", "Manual Strategy", "Tree Strategy Learner", "Q-Strategy Learner"],
    default=["Benchmark", "Tree Strategy Learner"]
)

# Advanced strategy parameters (collapsible sections)
if "Manual Strategy" in selected_strategies:
    with st.sidebar.expander("Manual Strategy Parameters"):
        # Get default values from config if available
        default_manual_params = {
            'window_size': 20,
            'rsi_window': 14,
            'buy_threshold': 0.02,
            'sell_threshold': -0.02,
            'position_size': 1000
        }
        
        # Override with values from config if available and enabled
        if use_configs and 'manual' in configs and 'manual_strategy' in configs['manual']:
            conf_manual = configs['manual']['manual_strategy']
            if 'window_size' in conf_manual:
                default_manual_params['window_size'] = conf_manual['window_size']
            if 'rsi_window' in conf_manual:
                default_manual_params['rsi_window'] = conf_manual['rsi_window']
            if 'buy_threshold' in conf_manual:
                default_manual_params['buy_threshold'] = conf_manual['buy_threshold']
            if 'sell_threshold' in conf_manual:
                default_manual_params['sell_threshold'] = conf_manual['sell_threshold']
            if 'position_size' in conf_manual:
                default_manual_params['position_size'] = conf_manual['position_size']
        
        manual_params = {
            'window_size': st.number_input("Technical Indicator Window Size", min_value=5, max_value=50, value=default_manual_params['window_size'], key="manual_window"),
            'rsi_window': st.number_input("RSI Window", min_value=5, max_value=30, value=default_manual_params['rsi_window'], key="manual_rsi"),
            'buy_threshold': st.number_input("Buy Threshold", min_value=0.01, max_value=0.1, value=default_manual_params['buy_threshold'], format="%.2f", key="manual_buy"),
            'sell_threshold': st.number_input("Sell Threshold", min_value=-0.1, max_value=-0.01, value=default_manual_params['sell_threshold'], format="%.2f", key="manual_sell"),
            'position_size': st.number_input("Position Size (Shares)", min_value=100, max_value=10000, value=default_manual_params['position_size'], step=100, key="manual_pos")
        }

if "Tree Strategy Learner" in selected_strategies:
    with st.sidebar.expander("Tree Strategy Parameters"):
        # Get default values from config if available
        default_tree_params = {
            'window_size': 20,
            'buy_threshold': 0.02,
            'sell_threshold': -0.02,
            'prediction_days': 5,
            'leaf_size': 5,
            'bags': 20,
            'position_size': 1000
        }
        
        # Override with values from config if available and enabled
        if use_configs and 'tree' in configs and 'tree_strategy_learner' in configs['tree']:
            conf_tree = configs['tree']['tree_strategy_learner']
            if 'window_size' in conf_tree:
                default_tree_params['window_size'] = conf_tree['window_size']
            if 'buy_threshold' in conf_tree:
                default_tree_params['buy_threshold'] = conf_tree['buy_threshold']
            if 'sell_threshold' in conf_tree:
                default_tree_params['sell_threshold'] = conf_tree['sell_threshold']
            if 'prediction_days' in conf_tree:
                default_tree_params['prediction_days'] = conf_tree['prediction_days']
            if 'leaf_size' in conf_tree:
                default_tree_params['leaf_size'] = conf_tree['leaf_size']
            if 'bags' in conf_tree:
                default_tree_params['bags'] = conf_tree['bags']
            if 'position_size' in conf_tree:
                default_tree_params['position_size'] = conf_tree['position_size']
        
        tree_params = {
            'window_size': st.number_input("Window Size", min_value=5, max_value=50, value=default_tree_params['window_size'], key="tree_window"),
            'buy_threshold': st.number_input("Buy Threshold", min_value=0.01, max_value=0.1, value=default_tree_params['buy_threshold'], format="%.2f", key="tree_buy"),
            'sell_threshold': st.number_input("Sell Threshold", min_value=-0.1, max_value=-0.01, value=default_tree_params['sell_threshold'], format="%.2f", key="tree_sell"),
            'prediction_days': st.number_input("Prediction Days Ahead", min_value=1, max_value=20, value=default_tree_params['prediction_days'], key="tree_pred"),
            'leaf_size': st.number_input("Leaf Size", min_value=1, max_value=20, value=default_tree_params['leaf_size'], key="tree_leaf"),
            'bags': st.number_input("Number of Bags", min_value=5, max_value=50, value=default_tree_params['bags'], key="tree_bags"),
            'position_size': st.number_input("Position Size (Shares)", min_value=100, max_value=10000, value=default_tree_params['position_size'], step=100, key="tree_pos")
        }

if "Q-Strategy Learner" in selected_strategies:
    with st.sidebar.expander("Q-Strategy Parameters"):
        # Get default values from config if available
        default_q_params = {
            'indicator_bins': 10,
            'window_size': 20,
            'rsi_window': 14,
            'stoch_window': 14,
            'cci_window': 20,
            'position_size': 1000,
            'max_iterations': 100,
            'use_bb': True,
            'use_rsi': True,
            'use_macd': True,
            'use_stoch': False,
            'use_cci': False,
            'momentum_periods': [3, 5, 10],
            'learning_rate': 0.2,
            'discount_factor': 0.9,
            'random_action_rate': 0.5,
            'random_action_decay': 0.99,
            'dyna_iterations': 10,
            'convergence_threshold': 0.1
        }
        
        # Override with values from config if available and enabled
        if use_configs and 'qlearn' in configs and 'q_strategy_learner' in configs['qlearn']:
            conf_q = configs['qlearn']['q_strategy_learner']
            
            for param in default_q_params:
                if param in conf_q:
                    default_q_params[param] = conf_q[param]
        
        # Basic parameters section
        st.sidebar.subheader("Basic Parameters")
        q_params = {
            'indicator_bins': st.number_input("Indicator Bins", min_value=5, max_value=20, value=default_q_params['indicator_bins'], key="q_bins"),
            'window_size': st.number_input("Window Size", min_value=5, max_value=50, value=default_q_params['window_size'], key="q_window"),
            'position_size': st.number_input("Position Size (Shares)", min_value=100, max_value=10000, value=default_q_params['position_size'], step=100, key="q_pos"),
            'max_iterations': st.number_input("Max Training Iterations", min_value=10, max_value=500, value=default_q_params['max_iterations'], step=10, key="q_iter"),
        }
        
        # Indicator selection section
        st.sidebar.subheader("Technical Indicators")
        q_params.update({
            'use_bb': st.checkbox("Use Bollinger Bands", value=default_q_params['use_bb'], key="q_use_bb"),
            'use_rsi': st.checkbox("Use RSI", value=default_q_params['use_rsi'], key="q_use_rsi"),
            'use_macd': st.checkbox("Use MACD", value=default_q_params['use_macd'], key="q_use_macd"),
            'use_stoch': st.checkbox("Use Stochastic Oscillator", value=default_q_params['use_stoch'], key="q_use_stoch"),
            'use_cci': st.checkbox("Use CCI", value=default_q_params['use_cci'], key="q_use_cci"),
        })
        
        # Show indicator-specific parameters if selected
        if q_params['use_rsi']:
            q_params['rsi_window'] = st.number_input("RSI Window", min_value=5, max_value=30, value=default_q_params['rsi_window'], key="q_rsi")
        
        if q_params['use_stoch']:
            q_params['stoch_window'] = st.number_input("Stochastic Window", min_value=5, max_value=30, value=default_q_params['stoch_window'], key="q_stoch")
        
        if q_params['use_cci']:
            q_params['cci_window'] = st.number_input("CCI Window", min_value=5, max_value=30, value=default_q_params['cci_window'], key="q_cci")
        
        # Momentum parameters
        st.sidebar.subheader("Momentum Parameters")
        use_momentum = st.checkbox("Use Momentum Indicators", value=len(default_q_params['momentum_periods']) > 0, key="q_use_momentum")
        if use_momentum:
            momentum_3 = st.checkbox("3-day Momentum", value=3 in default_q_params['momentum_periods'], key="q_mom_3")
            momentum_5 = st.checkbox("5-day Momentum", value=5 in default_q_params['momentum_periods'], key="q_mom_5")
            momentum_10 = st.checkbox("10-day Momentum", value=10 in default_q_params['momentum_periods'], key="q_mom_10")
            
            q_params['momentum_periods'] = [
                period for period, selected in [
                    (3, momentum_3), 
                    (5, momentum_5), 
                    (10, momentum_10)
                ] if selected
            ]
        else:
            q_params['momentum_periods'] = []
        
        # Q-Learning parameters section
        st.sidebar.subheader("Q-Learning Parameters")
        q_params.update({
            'learning_rate': st.number_input("Learning Rate", min_value=0.05, max_value=0.5, value=default_q_params['learning_rate'], format="%.2f", key="q_alpha"),
            'discount_factor': st.number_input("Discount Factor", min_value=0.5, max_value=1.0, value=default_q_params['discount_factor'], format="%.2f", key="q_gamma"),
            'random_action_rate': st.number_input("Initial Random Action Rate", min_value=0.1, max_value=1.0, value=default_q_params['random_action_rate'], format="%.2f", key="q_rar"),
            'random_action_decay': st.number_input("Random Action Decay Rate", min_value=0.9, max_value=1.0, value=default_q_params['random_action_decay'], format="%.3f", key="q_radr"),
            'dyna_iterations': st.number_input("Dyna-Q Planning Iterations", min_value=0, max_value=50, value=default_q_params['dyna_iterations'], key="q_dyna"),
            'convergence_threshold': st.number_input("Convergence Threshold", min_value=0.01, max_value=1.0, value=default_q_params['convergence_threshold'], format="%.2f", key="q_conv"),
        })

# Run button
run_button = st.sidebar.button("Run Strategies")

# Main content area
if run_button:
    if not selected_strategies:
        st.warning("Please select at least one strategy to run.")
    else:
        # Show a progress indicator
        with st.spinner("Running strategies..."):
            # Set up the figure for the performance chart
            results = {}
            
            # Run selected strategies
            if "Benchmark" in selected_strategies:
                benchmark_values = create_benchmark(
                    symbol=selected_symbol,
                    start_date=test_start,
                    end_date=test_end,
                    starting_value=starting_value,
                    commission=commission,
                    impact=impact
                )
                results["Benchmark"] = benchmark_values
            
            if "Manual Strategy" in selected_strategies:
                manual_portvals, manual_trades, manual_metrics = run_strategy(
                    strategy_type='manual',
                    symbol=selected_symbol,
                    train_start_date=train_start,
                    train_end_date=train_end,
                    test_start_date=test_start,
                    test_end_date=test_end,
                    starting_value=starting_value,
                    commission=commission,
                    impact=impact,
                    **manual_params
                )
                results["Manual Strategy"] = manual_portvals
            
            if "Tree Strategy Learner" in selected_strategies:
                tree_portvals, tree_trades, tree_metrics = run_strategy(
                    strategy_type='tree',
                    symbol=selected_symbol,
                    train_start_date=train_start,
                    train_end_date=train_end,
                    test_start_date=test_start,
                    test_end_date=test_end,
                    starting_value=starting_value,
                    commission=commission,
                    impact=impact,
                    **tree_params
                )
                results["Tree Strategy Learner"] = tree_portvals
            
            if "Q-Strategy Learner" in selected_strategies:
                q_portvals, q_trades, q_metrics = run_strategy(
                    strategy_type='qlearn',
                    symbol=selected_symbol,
                    train_start_date=train_start,
                    train_end_date=train_end,
                    test_start_date=test_start,
                    test_end_date=test_end,
                    starting_value=starting_value,
                    commission=commission,
                    impact=impact,
                    **q_params
                )
                results["Q-Strategy Learner"] = q_portvals
            
            # Display performance chart
            st.header("Strategy Performance Comparison")
            st.markdown(f"**Symbol: {selected_symbol}** | **Test Period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}**")
            
            fig, ax = plt.figure(figsize=(12, 6)), plt.gca()
            
            colors = {
                'Benchmark': 'black',
                'Manual Strategy': 'red',
                'Tree Strategy Learner': 'blue',
                'Q-Strategy Learner': 'green'
            }
            
            # Plot normalized portfolio values
            for strategy_name, portfolio_values in results.items():
                normalized_values = portfolio_values / portfolio_values.iloc[0]
                ax.plot(normalized_values.index, normalized_values, label=strategy_name, color=colors[strategy_name], linewidth=2)
            
            plt.title(f"{selected_symbol} Performance Comparison", fontsize=16)
            plt.xlabel("Date", fontsize=12)
            plt.ylabel("Normalized Portfolio Value", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc="best", fontsize=12)
            
            st.pyplot(fig)
            
            # Display performance metrics
            st.header("Performance Metrics")
            
            metrics_data = []
            
            for strategy_name, portfolio_values in results.items():
                metrics = calculate_metrics(portfolio_values)
                metrics_data.append({
                    "Strategy": strategy_name,
                    "Cumulative Return (%)": f"{metrics['cumulative_return'] * 100:.2f}%",
                    "Average Daily Return (%)": f"{metrics['average_daily_return'] * 100:.4f}%",
                    "Std Dev Daily Return (%)": f"{metrics['std_daily_return'] * 100:.4f}%",
                    "Sharpe Ratio": f"{metrics['sharpe_ratio']:.4f}",
                    "Final Portfolio Value": f"${metrics['final_value']:,.2f}"
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            # Display strategy parameters
            st.header("Strategy Parameters")
            
            for strategy_name in selected_strategies:
                if strategy_name == "Manual Strategy" and "Manual Strategy" in selected_strategies:
                    st.subheader("Manual Strategy Parameters")
                    st.json(manual_params)
                
                elif strategy_name == "Tree Strategy Learner" and "Tree Strategy Learner" in selected_strategies:
                    st.subheader("Tree Strategy Learner Parameters")
                    st.json(tree_params)
                
                elif strategy_name == "Q-Strategy Learner" and "Q-Strategy Learner" in selected_strategies:
                    st.subheader("Q-Strategy Learner Parameters")
                    st.json(q_params)
                    
            # Display trade statistics if available
            if len(selected_strategies) > 1:  # Only show if comparing strategies
                st.header("Trade Statistics")
                
                trade_stats = []
                
                if "Manual Strategy" in selected_strategies:
                    trade_count = (manual_trades != 0).sum().sum()
                    trade_stats.append({
                        "Strategy": "Manual Strategy",
                        "Number of Trades": trade_count,
                        "Trades per Month": f"{trade_count / (len(manual_trades) / 21):.2f}"
                    })
                
                if "Tree Strategy Learner" in selected_strategies:
                    trade_count = (tree_trades != 0).sum().sum()
                    trade_stats.append({
                        "Strategy": "Tree Strategy Learner",
                        "Number of Trades": trade_count,
                        "Trades per Month": f"{trade_count / (len(tree_trades) / 21):.2f}"
                    })
                
                if "Q-Strategy Learner" in selected_strategies:
                    trade_count = (q_trades != 0).sum().sum()
                    trade_stats.append({
                        "Strategy": "Q-Strategy Learner",
                        "Number of Trades": trade_count,
                        "Trades per Month": f"{trade_count / (len(q_trades) / 21):.2f}"
                    })
                
                if trade_stats:
                    st.dataframe(pd.DataFrame(trade_stats), hide_index=True, use_container_width=True)
else:
    st.info("Select your strategy parameters and click 'Run Strategies' to see results.")
    
    # Show data preview
    st.header(f"Data Preview for {selected_symbol}")
    
    try:
        # Get a sample of the price data
        preview_start = dt.datetime(2010, 1, 1)  
        preview_end = dt.datetime(2010, 1, 31)  # Just show a month
        price_data = get_data([selected_symbol], pd.date_range(preview_start, preview_end))
        
        # Plot the price data
        fig, ax = plt.figure(figsize=(10, 5)), plt.gca()
        ax.plot(price_data.index, price_data[selected_symbol], label=selected_symbol, linewidth=2)
        plt.title(f"{selected_symbol} Price (Jan 2010)", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price ($)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        st.pyplot(fig)
        
        # Display the first few rows of data
        st.subheader("Price Data Sample")
        st.dataframe(price_data.head(10), use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data for {selected_symbol}: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("### About TradingStrategist")
st.markdown("""
    TradingStrategist is a comprehensive machine learning-based trading strategy platform that implements and evaluates 
    multiple algorithmic trading approaches. This Streamlit application provides a user-friendly interface 
    to interact with the underlying framework.

    **Disclaimer**: This tool is for educational purposes only. Trading involves risk. Past performance is not indicative of future results.
""")