"""
Experiment 1: Comparison of Manual Strategy vs ML Strategy

This experiment compares the performance of a rule-based Manual Strategy
against a machine learning-based Strategy Learner.
"""

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

# Update import to use TreeStrategyLearner instead of StrategyLearner
from TradingStrategist.models.ManualStrategy import ManualStrategy
from TradingStrategist.models.TreeStrategyLearner import TreeStrategyLearner  # Updated import
from TradingStrategist.data.loader import get_data
from TradingStrategist.simulation.market_sim import compute_portvals, assess_portfolio
from TradingStrategist.utils.helpers import get_output_path, load_config


def run_strategy_comparison(symbol, sd, ed, sv=100000, commission=9.95, impact=0.005, 
                           sl_params=None, ms_params=None, config=None):
    """
    Run and compare Manual Strategy against Strategy Learner.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol to trade
    sd : datetime
        Start date
    ed : datetime
        End date
    sv : int
        Starting portfolio value
    commission : float
        Commission per trade
    impact : float
        Market impact per trade
    sl_params : dict
        Parameters for strategy learner
    ms_params : dict
        Parameters for manual strategy
    config : dict
        Full configuration dictionary
        
    Returns:
    --------
    dict
        Dictionary containing results for each strategy
    """
    results = {}
    
    # Get position size from config (default to 1000)
    position_size = 1000
    if config and 'portfolio' in config and 'max_positions' in config['portfolio']:
        position_size = config['portfolio']['max_positions']
    
    # Create and run Manual Strategy with parameters
    if ms_params is None:
        ms_params = {}
        
    manual_strategy = ManualStrategy(verbose=False, **ms_params)
    ms_trades = manual_strategy.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    
    # Pass config to compute_portvals
    ms_portvals = compute_portvals(
        orders_df=ms_trades, 
        start_val=sv, 
        commission=commission, 
        impact=impact,
        config=config
    )
    
    results["Manual Strategy"] = {
        "trades": ms_trades,
        "portvals": ms_portvals,
    }
    
    # Create and run Strategy Learner with parameters from config
    if sl_params is None:
        sl_params = {}
    
    # Create TreeStrategyLearner with proper parameters
    strategy_learner = TreeStrategyLearner(
        verbose=False, 
        impact=impact,
        commission=commission,
        **{k: v for k, v in sl_params.items() if k not in ['verbose', 'impact', 'commission']}
    )
    
    strategy_learner.addEvidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    sl_trades = strategy_learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    
    # Pass config to compute_portvals
    sl_portvals = compute_portvals(
        orders_df=sl_trades, 
        start_val=sv, 
        commission=commission, 
        impact=impact,
        config=config
    )
    
    results["ML Strategy"] = {
        "trades": sl_trades,
        "portvals": sl_portvals,
    }
    
    # Create and run benchmark with configurable position size
    benchmark_trades = pd.DataFrame(index=ms_trades.index)
    benchmark_trades[symbol] = 0
    benchmark_trades.iloc[0] = position_size  # Use configurable position size
    
    # Pass config to compute_portvals
    benchmark_portvals = compute_portvals(
        orders_df=benchmark_trades, 
        start_val=sv, 
        commission=commission, 
        impact=impact,
        config=config
    )
    
    results["Benchmark"] = {
        "trades": benchmark_trades,
        "portvals": benchmark_portvals,
    }
    
    return results


def plot_strategy_comparison(results, title, filename, output_prefix, config=None):
    """
    Plot comparison of different strategies.
    
    Parameters:
    -----------
    results : dict
        Dictionary of strategy results
    title : str
        Plot title
    filename : str
        Output file name
    output_prefix : str
        Prefix for output filename
    config : dict, optional
        Configuration dictionary
    """
    # Get figure size from config if available
    fig_size = (10, 6)
    if config and 'visualization' in config and 'fig_size' in config['visualization']:
        fig_size = tuple(config['visualization']['fig_size'])
        
    plt.figure(figsize=fig_size)
    
    # Get colors from config if available
    colors = {
        "Manual Strategy": "red",
        "ML Strategy": "blue",
        "Benchmark": "purple"
    }
    
    if config and 'visualization' in config and 'colors' in config['visualization']:
        viz_colors = config['visualization']['colors']
        if 'manual_strategy' in viz_colors:
            colors["Manual Strategy"] = viz_colors['manual_strategy']
        if 'ml_strategy' in viz_colors:
            colors["ML Strategy"] = viz_colors['ml_strategy']
        if 'benchmark' in viz_colors:
            colors["Benchmark"] = viz_colors['benchmark']
    
    for strategy_name, result in results.items():
        portvals = result["portvals"]
        # Normalize portfolio values
        norm_vals = portvals / portvals.iloc[0]
        plt.plot(norm_vals, label=strategy_name, color=colors[strategy_name])
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid(True)
    
    output_path = get_output_path()
    plt.savefig(os.path.join(output_path, f"{output_prefix}_{filename}"))
    plt.close()


def calc_strategy_stats(results, config=None):
    """
    Calculate performance statistics for all strategies.
    
    Parameters:
    -----------
    results : dict
        Dictionary of strategy results
    config : dict, optional
        Configuration dictionary
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing performance metrics
    """
    stats_list = []
    
    # Get sampling frequency from config (default to 252)
    sampling_freq = 252
    if config and 'performance' in config and 'sampling_frequency' in config['performance']:
        sampling_freq = config['performance']['sampling_frequency']
    
    for strategy_name, result in results.items():
        portvals = result["portvals"]
        trades = result["trades"]
        
        # Calculate daily returns
        daily_returns = portvals.pct_change().dropna()
        
        # Calculate metrics
        cum_return = (portvals.iloc[-1] / portvals.iloc[0]) - 1
        avg_daily_ret = daily_returns.mean()
        std_daily_ret = daily_returns.std()
        sharpe_ratio = np.sqrt(sampling_freq) * avg_daily_ret / std_daily_ret
        trade_count = (trades != 0).sum().sum()
        
        stats_list.append({
            "Strategy": strategy_name,
            "Cumulative Return": cum_return.values[0],
            "Average Daily Return": avg_daily_ret.values[0],
            "Std Dev Daily Return": std_daily_ret.values[0],
            "Sharpe Ratio": sharpe_ratio.values[0],
            "Trade Count": trade_count
        })
    
    return pd.DataFrame(stats_list).set_index("Strategy")


def run_experiment_from_config(config_path):
    """
    Run Experiment 1 using parameters from a config file.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    """
    # Load config
    config = load_config(config_path)
    
    # Extract parameters
    symbol = config['data']['symbol']
    output_prefix = config['experiment']['output_prefix']
    
    # Get date ranges
    if 'training' in config['data'] and 'testing' in config['data']:
        train_start = dt.datetime.strptime(config['data']['training']['start_date'], '%Y-%m-%d')
        train_end = dt.datetime.strptime(config['data']['training']['end_date'], '%Y-%m-%d')
        test_start = dt.datetime.strptime(config['data']['testing']['start_date'], '%Y-%m-%d')
        test_end = dt.datetime.strptime(config['data']['testing']['end_date'], '%Y-%m-%d')
    else:
        # If not specified, use in-sample data for both
        train_start = dt.datetime.strptime(config['data']['in_sample']['start_date'], '%Y-%m-%d')
        train_end = dt.datetime.strptime(config['data']['in_sample']['end_date'], '%Y-%m-%d')
        test_start = train_start
        test_end = train_end
    
    # Get portfolio parameters
    starting_value = config['portfolio']['starting_value']
    
    # Get trading parameters
    trading_config = config.get('trading', {})
    commission = trading_config.get('commission', 9.95)
    impact = trading_config.get('impact', 0.005)
    
    # Get strategy learner parameters - look for dt_strategy first, fall back to strategy_learner
    if 'dt_strategy' in config:
        sl_params = config['dt_strategy']
    elif 'strategy_learner' in config:
        print("Warning: 'strategy_learner' config section is deprecated, use 'dt_strategy' instead")
        sl_params = config['strategy_learner']
    else:
        sl_params = {}
    
    # Get manual strategy parameters
    ms_params = config.get('manual_strategy', {})
    
    print(f"Running Experiment 1: Manual Strategy vs ML Strategy on {symbol}")
    print(f"Training period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
    print(f"Testing period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
    
    # Run experiment with all parameters from config
    results = run_strategy_comparison(
        symbol=symbol,
        sd=test_start,
        ed=test_end,
        sv=starting_value,
        commission=commission,
        impact=impact,
        sl_params=sl_params,
        ms_params=ms_params,
        config=config  # Pass full config
    )
    
    # Plot results
    plot_strategy_comparison(
        results,
        f"Experiment 1: Manual Strategy vs ML Strategy ({symbol})",
        "comparison.png",
        output_prefix,
        config=config  # Pass config for visualization
    )
    
    # Calculate and print statistics
    stats_df = calc_strategy_stats(results, config=config)
    print("\nPerformance Statistics:")
    print("-" * 80)
    print(stats_df)


def main():
    """Main function to run experiment from config file."""
    parser = argparse.ArgumentParser(description='Run Experiment 1: Manual Strategy vs ML Strategy')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Run experiment with config
    run_experiment_from_config(args.config)


if __name__ == "__main__":
    main()