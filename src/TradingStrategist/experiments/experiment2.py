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
import time
from pathlib import Path
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

from TradingStrategist.data.loader import get_data
from TradingStrategist.models.TreeStrategyLearner import TreeStrategyLearner
from TradingStrategist.simulation.market_sim import compute_portvals
from TradingStrategist.utils.helpers import get_output_path, load_config


def print_section(title):
    """Print a colored section header"""
    print(f"\n{Fore.CYAN}{Style.BRIGHT}" + "="*80 + f"{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}    {title}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "="*80 + f"{Style.RESET_ALL}")


def print_subsection(title):
    """Print a colored subsection header"""
    print(f"\n{Fore.YELLOW}{Style.BRIGHT}>> {title}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{Style.BRIGHT}" + "-"*60 + f"{Style.RESET_ALL}")


def print_status(message):
    """Print a status message"""
    print(f"{Fore.GREEN}● {message}{Style.RESET_ALL}")


def print_file_saved(file_path):
    """Print a message when a file is saved"""
    print(f"{Fore.MAGENTA}✓ SAVED: {file_path}{Style.RESET_ALL}")


def run_impact_experiment(symbol, sd, ed, sv=100000, commission=9.95, impact_values=None, 
                         sl_params=None, config=None):
    """
    Run experiments with different market impact values.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol to test
    sd : datetime
        Start date for testing
    ed : datetime
        End date for testing
    sv : int, optional
        Starting portfolio value, default 100000
    commission : float, optional
        Commission cost per trade, default 9.95
    impact_values : list, optional
        List of market impact values to test
    sl_params : dict, optional
        Parameters for Strategy Learner
    config : dict, optional
        Configuration dictionary
        
    Returns:
    --------
    tuple
        (Dict of portfolio values for each impact, Dict of trade counts)
    """
    if impact_values is None:
        impact_values = [0.0, 0.005, 0.01, 0.02, 0.04]
        
    if sl_params is None:
        sl_params = {}
        
    results = {}
    trade_counts = {}
    
    # Get position size from config (default to 1000)
    position_size = 1000
    if config and 'portfolio' in config and 'max_positions' in config['portfolio']:
        position_size = config['portfolio']['max_positions']
    
    print_subsection("1/3 - Preparing Benchmark Strategy")
    print_status(f"Creating benchmark (buy and hold) strategy for {symbol}...")
    
    # Get benchmark (buy and hold shares)
    dates = pd.date_range(sd, ed)
    
    # Pass config to get_data
    prices = get_data([symbol], dates, addSPY=True, colname="Adj Close", config=config)
    prices = prices[prices.index.isin(get_data([symbol], dates, config=config).index)]  # Get rid of NaN rows
    
    benchmark_trades = pd.DataFrame(index=prices.index)
    benchmark_trades[symbol] = 0
    benchmark_trades.iloc[0] = position_size  # Use configurable position size
    
    print_status(f"Simulating benchmark portfolio...")
    benchmark_portvals = compute_portvals(
        orders=benchmark_trades, 
        start_val=sv, 
        commission=commission, 
        impact=0.0
    )
    
    results["Benchmark"] = benchmark_portvals
    trade_counts["Benchmark"] = 1  # Just initial purchase
    print_status(f"Benchmark portfolio calculated with initial purchase of {position_size} shares")
    
    print_subsection(f"2/3 - Testing Impact Values")
    print_status(f"Running strategy with {len(impact_values)} different impact values: {impact_values}")

    # Run strategy with different impact values
    for i, impact in enumerate(impact_values):
        impact_label = f"Impact {impact}"
        print(f"\n{Fore.BLUE}[{i+1}/{len(impact_values)}] Testing impact value: {impact}{Style.RESET_ALL}")
        
        # Train the learner with current impact value and config parameters
        print_status(f"Training TreeStrategyLearner with impact={impact}, commission=${commission}...")
        start_time = time.time()
        learner = TreeStrategyLearner(
            verbose=False, 
            impact=impact,
            commission=commission,
            **{k: v for k, v in sl_params.items() if k not in ['verbose', 'impact', 'commission']}
        )
        
        learner.addEvidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
        train_time = time.time() - start_time
        print_status(f"Model training completed in {train_time:.2f} seconds")
        
        # Generate trades
        print_status(f"Generating trades from trained model...")
        start_time = time.time()
        trades = learner.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
        test_time = time.time() - start_time
        trade_count = (trades != 0).sum().sum()
        print_status(f"Generated {trade_count} trades in {test_time:.2f} seconds")
        
        # Compute portfolio values using same impact for market sim
        print_status(f"Simulating market with trades (impact={impact})...")
        start_time = time.time()
        portvals = compute_portvals(
            orders=trades, 
            start_val=sv, 
            commission=commission, 
            impact=impact
        )
        sim_time = time.time() - start_time
        
        # Calculate cumulative return
        cum_return = (portvals.iloc[-1] / portvals.iloc[0] - 1)[0] * 100
        
        # Store results
        results[impact_label] = portvals
        trade_counts[impact_label] = trade_count
        print_status(f"Portfolio simulation completed in {sim_time:.2f} seconds")
        print_status(f"Impact {impact} results: {trade_count} trades, {cum_return:.2f}% return")
    
    print_status(f"All impact value tests completed")
    return results, trade_counts


def plot_impact_results(results, trade_counts, title, filename, output_prefix, config=None):
    """
    Plot portfolio performance for different impact values.
    
    Parameters:
    -----------
    results : dict
        Dictionary of portfolio values for each impact value
    trade_counts : dict
        Dictionary of trade counts for each impact value
    title : str
        Plot title
    filename : str
        Output file name
    output_prefix : str
        Prefix for output filename
    config : dict, optional
        Configuration dictionary
    """
    print_subsection("3/3 - Generating Results")
    
    # Create output directory if it doesn't exist
    output_path = get_output_path()
    output_file = os.path.join(output_path, f"{output_prefix}_{filename}")
    
    print_status(f"Creating performance comparison chart...")
    
    # Plot performance
    plt.figure(figsize=(12, 8))
    
    # Plot normalized portfolio values
    ax1 = plt.subplot(211)
    
    for label, portvals in results.items():
        # Normalize to initial portfolio value
        norm_portvals = portvals / portvals.iloc[0]
        
        # Determine line style and color
        if label == "Benchmark":
            linestyle = "--"
            linewidth = 2
            color = "black"
        else:
            linestyle = "-"
            linewidth = 1.5
            # Define colors based on impact value
            impact = float(label.split()[-1])
            # Use a gradient from green (low impact) to red (high impact)
            color = plt.cm.RdYlGn_r(impact / 0.04 if impact > 0 else 0)
        
        ax1.plot(
            norm_portvals.index,
            norm_portvals,
            label=f"{label} ({trade_counts[label]} trades)",
            linewidth=linewidth,
            linestyle=linestyle,
            color=color
        )
    
    ax1.set_title(title)
    ax1.set_ylabel("Normalized Portfolio Value")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # Plot final returns as bar chart
    ax2 = plt.subplot(212)
    
    # Calculate returns
    returns = []
    impact_labels = []
    
    for label, portvals in results.items():
        returns.append((portvals.iloc[-1] / portvals.iloc[0] - 1)[0] * 100)
        impact_labels.append(label)
    
    # Map labels to colors
    colors = []
    for label in impact_labels:
        if label == "Benchmark":
            colors.append("black")
        else:
            impact = float(label.split()[-1])
            colors.append(plt.cm.RdYlGn_r(impact / 0.04 if impact > 0 else 0))
            
    # Create bar chart
    bars = ax2.bar(impact_labels, returns, color=colors)
    
    # Add value labels on bars
    for bar, count in zip(bars, [trade_counts[label] for label in impact_labels]):
        height = bar.get_height()
        vertical_offset = 3 if height > 0 else -10
        ax2.annotate(
            f"{height:.1f}%\n{count} trades",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, vertical_offset),
            textcoords="offset points",
            ha='center', va='bottom'
        )
        
    ax2.set_title("Cumulative Returns by Impact Value")
    ax2.set_ylabel("Cumulative Return (%)")
    ax2.set_xlabel("Market Impact Value")
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()
    
    print_file_saved(output_file)
    
    # Create and save a CSV with the results
    stats_file = os.path.join(output_path, f"{output_prefix}_impact_results.csv")
    
    # Calculate statistics for each impact value
    stats = []
    
    for label, portvals in results.items():
        daily_returns = portvals.pct_change().dropna()
        cum_return = (portvals.iloc[-1] / portvals.iloc[0] - 1)[0]
        avg_daily_return = daily_returns.mean()[0]
        std_daily_return = daily_returns.std()[0]
        sharpe_ratio = np.sqrt(252) * avg_daily_return / std_daily_return
        
        # Extract impact value from label (or use 'N/A' for benchmark)
        impact_value = label.split()[-1] if label != "Benchmark" else "N/A"
        
        stats.append({
            "Strategy": label,
            "Impact": impact_value,
            "Trade Count": trade_counts[label],
            "Cumulative Return": cum_return,
            "Avg Daily Return": avg_daily_return,
            "Std Daily Return": std_daily_return,
            "Sharpe Ratio": sharpe_ratio
        })
    
    # Create DataFrame and save to CSV
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(stats_file, index=False)
    
    print_file_saved(stats_file)
    
    # Print statistics
    print_subsection("IMPACT EXPERIMENT RESULTS")
    print("\nTrade counts by impact value:")
    for label, count in trade_counts.items():
        impact_str = label if label == "Benchmark" else f"Impact = {label.split()[-1]}"
        print(f"  {impact_str}: {count} trades")
    
    print("\nCumulative returns by impact value:")
    for label, portvals in results.items():
        cum_return = (portvals.iloc[-1] / portvals.iloc[0] - 1)[0] * 100
        impact_str = label if label == "Benchmark" else f"Impact = {label.split()[-1]}"
        print(f"  {impact_str}: {cum_return:.2f}%")
    
    print(f"\nDetailed statistics saved to: {stats_file}")
    print(f"Performance plot saved to: {output_file}")


def run_experiment(config_path):
    """
    Run experiment 2 with impact values.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Extract experiment parameters
    experiment_config = config.get('experiment', {})
    data_config = config.get('data', {})
    portfolio_config = config.get('portfolio', {})
    dt_config = config.get('dt_strategy', {})
    trading_config = config.get('trading', {})
    
    # Get experiment name and prefix for output files
    experiment_name = experiment_config.get('name', 'Experiment 2')
    output_prefix = experiment_config.get('output_prefix', 'exp2')
    
    # Get symbol and date range
    symbol = data_config.get('symbol', 'JPM')
    start_date = dt.datetime.strptime(data_config.get('start_date', '2008-01-01'), '%Y-%m-%d')
    end_date = dt.datetime.strptime(data_config.get('end_date', '2009-12-31'), '%Y-%m-%d')
    
    # Get starting value
    starting_value = portfolio_config.get('starting_value', 100000)
    
    # Get commission
    commission = trading_config.get('commission', 9.95)
    
    # Get impact values to test
    impact_values = trading_config.get('impact_values', [0.0, 0.005, 0.01, 0.02, 0.04])
    
    # Print experiment setup
    print_section(f"EXPERIMENT 2: {experiment_name}")
    
    print_status(f"Testing impact of transaction costs on ML strategy")
    print_status(f"Symbol: {symbol}")
    print_status(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print_status(f"Starting portfolio value: ${starting_value:,.2f}")
    print_status(f"Commission: ${commission:.2f}")
    print_status(f"Impact values to test: {impact_values}")
    
    # Run experiment
    results, trade_counts = run_impact_experiment(
        symbol=symbol,
        sd=start_date,
        ed=end_date,
        sv=starting_value,
        commission=commission,
        impact_values=impact_values,
        sl_params=dt_config,
        config=config
    )
    
    # Format date range for filename
    date_str = start_date.strftime('%Y%m%d') + '_' + end_date.strftime('%Y%m%d')
    filename = f"{symbol}_impact_comparison_{date_str}.png"
    
    # Plot results
    plot_impact_results(
        results=results,
        trade_counts=trade_counts,
        title=f"Impact of Market Impact on Strategy Performance ({symbol})",
        filename=filename,
        output_prefix=output_prefix,
        config=config
    )
    
    print(f"\n{Fore.GREEN}{Style.BRIGHT}Experiment completed successfully!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "="*80 + f"{Style.RESET_ALL}")


def main():
    """Run the experiment."""
    parser = argparse.ArgumentParser(description='Run Experiment 2: Impact of Transaction Costs')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    run_experiment(args.config)


if __name__ == "__main__":
    main()