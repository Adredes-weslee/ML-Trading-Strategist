"""
Experiment 2: Impact of Transaction Costs

This experiment explores how different market impact values affect 
the performance of the Strategy Learner algorithm.
"""

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

from TradingStrategist.data.loader import get_data
from TradingStrategist.models.TreeStrategyLearner import TreeStrategyLearner  # Updated import
from TradingStrategist.simulation.market_sim import compute_portvals
from TradingStrategist.utils.helpers import get_output_path, load_config


def run_impact_experiment(symbol, sd, ed, sv=100000, commission=9.95, impact_values=None, 
                         sl_params=None, config=None):
    """
    Run experiment with different impact values.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol to trade
    sd : datetime
        Start date for analysis
    ed : datetime
        End date for analysis
    sv : int
        Starting portfolio value
    commission : float
        Commission per trade
    impact_values : list
        List of impact values to test
    sl_params : dict
        Parameters for strategy learner
    config : dict, optional
        Full configuration dictionary
        
    Returns:
    --------
    dict
        Dictionary of portfolio values for each impact value
    dict
        Dictionary of trade counts for each impact value
    """
    if impact_values is None:
        impact_values = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
    
    if sl_params is None:
        sl_params = {}
        
    results = {}
    trade_counts = {}
    
    # Get position size from config (default to 1000)
    position_size = 1000
    if config and 'portfolio' in config and 'max_positions' in config['portfolio']:
        position_size = config['portfolio']['max_positions']
    
    # Get benchmark (buy and hold shares)
    dates = pd.date_range(sd, ed)
    
    # Pass config to get_data
    prices = get_data([symbol], dates, addSPY=True, colname="Adj Close", config=config)
    prices = prices[prices.index.isin(get_data([symbol], dates, config=config).index)]  # Get rid of NaN rows
    
    benchmark_trades = pd.DataFrame(index=prices.index)
    benchmark_trades[symbol] = 0
    benchmark_trades.iloc[0] = position_size  # Use configurable position size
    
    # Pass config to compute_portvals
    benchmark_portvals = compute_portvals(
        orders_df=benchmark_trades, 
        start_val=sv, 
        commission=commission, 
        impact=0.0,
        config=config
    )
    
    results["Benchmark"] = benchmark_portvals
    trade_counts["Benchmark"] = 1  # Just initial purchase
    
    # Run strategy with different impact values
    for impact in impact_values:
        # Train the learner with current impact value and config parameters
        learner = TreeStrategyLearner(
            verbose=False, 
            impact=impact,
            commission=commission,
            **{k: v for k, v in sl_params.items() if k not in ['verbose', 'impact', 'commission']}
        )
        
        learner.addEvidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
        
        # Generate trades
        trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        
        # Compute portfolio values using same impact for market sim
        portvals = compute_portvals(
            orders_df=trades, 
            start_val=sv, 
            commission=commission, 
            impact=impact,
            config=config
        )
        
        # Store results
        impact_label = f"Impact {impact}"
        results[impact_label] = portvals
        trade_counts[impact_label] = (trades != 0).sum().sum()
    
    return results, trade_counts


def plot_impact_results(results, title, filename, output_prefix, config=None):
    """
    Plot portfolio performance for different impact values.
    
    Parameters:
    -----------
    results : dict
        Dictionary of portfolio values for each impact value
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
    colors = ['purple', 'red', 'blue', 'green', 'orange', 'brown', 'black']
    if config and 'visualization' in config and 'impact_colors' in config['visualization']:
        colors = config['visualization']['impact_colors']
        
    color_idx = 0
    
    for label, portvals in results.items():
        # Normalize portfolio values
        norm_vals = portvals / portvals.iloc[0]
        plt.plot(norm_vals, label=label, color=colors[color_idx % len(colors)])
        color_idx += 1
    
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    plt.grid(True)
    
    output_path = get_output_path()
    plt.savefig(os.path.join(output_path, f"{output_prefix}_{filename}"))
    plt.close()


def plot_impact_vs_trades(trade_counts, impact_values, filename, output_prefix, config=None):
    """
    Plot impact values against number of trades.
    
    Parameters:
    -----------
    trade_counts : dict
        Dictionary of trade counts for each impact value
    impact_values : list
        List of impact values used
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
    
    # Extract trade counts and impacts (skip benchmark)
    impacts = impact_values
    trades = [trade_counts[f"Impact {impact}"] for impact in impact_values]
    
    # Get marker and line style from config if available
    marker = 'o'
    linestyle = '-'
    color = 'blue'
    
    if config and 'visualization' in config:
        viz_config = config['visualization']
        if 'marker' in viz_config:
            marker = viz_config['marker']
        if 'linestyle' in viz_config:
            linestyle = viz_config['linestyle']
        if 'trade_count_color' in viz_config:
            color = viz_config['trade_count_color']
    
    plt.plot(impacts, trades, marker=marker, linestyle=linestyle, color=color)
    plt.title("Number of Trades vs. Impact")
    plt.xlabel("Market Impact")
    plt.ylabel("Number of Trades")
    plt.grid(True)
    
    output_path = get_output_path()
    plt.savefig(os.path.join(output_path, f"{output_prefix}_{filename}"))
    plt.close()


def calc_all_stats(results, config=None):
    """
    Calculate performance statistics for all impact values.
    
    Parameters:
    -----------
    results : dict
        Dictionary of portfolio values for each impact value
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
    
    for label, portvals in results.items():
        # Calculate daily returns
        daily_returns = portvals.pct_change().dropna()
        
        # Calculate metrics
        cum_return = (portvals.iloc[-1] / portvals.iloc[0]) - 1
        avg_daily_ret = daily_returns.mean()
        std_daily_ret = daily_returns.std()
        sharpe_ratio = np.sqrt(sampling_freq) * avg_daily_ret / std_daily_ret
        
        stats_list.append({
            "Strategy": label,
            "Cumulative Return": cum_return.values[0],
            "Average Daily Return": avg_daily_ret.values[0],
            "Std Dev Daily Return": std_daily_ret.values[0],
            "Sharpe Ratio": sharpe_ratio.values[0]
        })
    
    return pd.DataFrame(stats_list).set_index("Strategy")


def run_experiment_from_config(config_path):
    """
    Run Experiment 2 using parameters from a config file.
    
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
    if 'training' in config['data']:
        start_date = dt.datetime.strptime(config['data']['training']['start_date'], '%Y-%m-%d')
        end_date = dt.datetime.strptime(config['data']['training']['end_date'], '%Y-%m-%d')
    else:
        start_date = dt.datetime.strptime(config['data']['in_sample']['start_date'], '%Y-%m-%d')
        end_date = dt.datetime.strptime(config['data']['in_sample']['end_date'], '%Y-%m-%d')
    
    # Get portfolio parameters
    starting_value = config['portfolio']['starting_value']
    
    # Get trading parameters
    trading_config = config['trading']
    commission = trading_config.get('commission', 9.95)
    
    # Check for impacts array or impacts field
    if 'impacts' in trading_config:
        impact_values = trading_config['impacts']
    elif 'impact_values' in trading_config:
        impact_values = trading_config['impact_values']
    else:
        impact_values = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
    
    # Get strategy learner parameters - look for dt_strategy first, fall back to strategy_learner
    if 'dt_strategy' in config:
        sl_params = config['dt_strategy']
    elif 'strategy_learner' in config:
        print("Warning: 'strategy_learner' config section is deprecated, use 'dt_strategy' instead")
        sl_params = config['strategy_learner']
    else:
        sl_params = {}
    
    print(f"Running Experiment 2: Impact of Transaction Costs on {symbol}")
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Testing impact values: {impact_values}")
    
    # Run experiment with all parameters from config
    results, trade_counts = run_impact_experiment(
        symbol=symbol,
        sd=start_date,
        ed=end_date,
        sv=starting_value,
        commission=commission,
        impact_values=impact_values,
        sl_params=sl_params,
        config=config  # Pass full config
    )
    
    # Plot portfolio performance
    plot_impact_results(
        results,
        f"Experiment 2: Strategy Performance with Different Impact Values ({symbol})",
        "performance.png",
        output_prefix,
        config=config  # Pass config for visualization
    )
    
    # Plot impact vs trades
    plot_impact_vs_trades(
        trade_counts,
        impact_values,
        "impact_vs_trades.png",
        output_prefix,
        config=config  # Pass config for visualization
    )
    
    # Calculate and print statistics with config
    stats_df = calc_all_stats(results, config=config)
    print("\nPerformance Statistics for Different Impact Values:")
    print("-" * 80)
    print(stats_df)
    
    # Print trade counts
    print("\nNumber of Trades for Different Impact Values:")
    print("-" * 50)
    for label, count in trade_counts.items():
        print(f"{label}: {count} trades")


def main():
    """Main function to run experiment from config file."""
    parser = argparse.ArgumentParser(description='Run Experiment 2: Impact of Transaction Costs')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Run experiment with config
    run_experiment_from_config(args.config)


if __name__ == "__main__":
    main()