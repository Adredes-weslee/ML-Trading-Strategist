"""
Manual Strategy Evaluation - Evaluates rule-based trading strategy against market benchmark

This module evaluates the performance of a rule-based manual trading strategy by:
1. Running the ManualStrategy against a buy & hold benchmark
2. Plotting comparative performance
3. Calculating key performance statistics
"""

import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

from TradingStrategist.models.ManualStrategy import ManualStrategy
from TradingStrategist.data.loader import get_data
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


def plot_strategy_comparison(trades, portvals, benchvals, title, filename, config=None):
    """
    Plot manual strategy performance against benchmark.
    
    Parameters:
    -----------
    trades : pandas.DataFrame
        DataFrame containing trade decisions
    portvals : pandas.DataFrame
        Portfolio values for manual strategy
    benchvals : pandas.DataFrame
        Portfolio values for benchmark
    title : str
        Plot title
    filename : str
        Output file path for saving plot
    config : dict, optional
        Configuration dictionary with visualization parameters
    """
    norm_ms = portvals / portvals.iloc[0]
    norm_bench = benchvals / benchvals.iloc[0]

    # Get figure size and colors from config
    fig_size = (10, 6)
    strategy_color = "red"
    benchmark_color = "purple"
    
    if config and 'visualization' in config:
        viz_config = config['visualization']
        if 'fig_size' in viz_config:
            fig_size = tuple(viz_config['fig_size'])
        if 'colors' in viz_config:
            colors = viz_config['colors']
            if 'manual_strategy' in colors:
                strategy_color = colors['manual_strategy']
            if 'benchmark' in colors:
                benchmark_color = colors['benchmark']

    plt.figure(figsize=fig_size)
    plt.plot(norm_ms, label="Manual Strategy", color=strategy_color)
    plt.plot(norm_bench, label="Benchmark", color=benchmark_color)

    # Configure trade markers from config
    buy_color = 'blue'
    sell_color = 'black'
    line_style = '--'
    alpha = 0.5
    
    if config and 'visualization' in config and 'trade_markers' in config['visualization']:
        markers = config['visualization']['trade_markers']
        if 'buy_color' in markers:
            buy_color = markers['buy_color']
        if 'sell_color' in markers:
            sell_color = markers['sell_color']
        if 'line_style' in markers:
            line_style = markers['line_style']
        if 'alpha' in markers:
            alpha = markers['alpha']

    # Add vertical lines for trades
    symbol = trades.columns[0]
    for i in range(len(trades)):
        if trades.iloc[i, 0] > 0:  # Buy signal
            plt.axvline(trades.index[i], color=buy_color, linestyle=line_style, alpha=alpha)
        elif trades.iloc[i, 0] < 0:  # Sell signal
            plt.axvline(trades.index[i], color=sell_color, linestyle=line_style, alpha=alpha)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend()
    
    # Ensure output directory exists
    output_path = get_output_path()
    file_path = os.path.join(output_path, filename)
    plt.savefig(file_path)
    plt.close()
    print_file_saved(file_path)


def calc_stats(portvals, benchvals, config=None):
    """
    Calculate performance statistics for strategy vs benchmark.
    
    Parameters:
    -----------
    portvals : pandas.DataFrame
        Portfolio values for strategy
    benchvals : pandas.DataFrame
        Portfolio values for benchmark
    config : dict, optional
        Configuration dictionary with performance parameters
        
    Returns:
    --------
    dict
        Dictionary containing performance metrics
    """
    # Calculate daily returns
    port_daily_rets = portvals.pct_change().dropna()
    bench_daily_rets = benchvals.pct_change().dropna()
    
    # Get sampling frequency from config (default to 252)
    sampling_freq = 252
    if config and 'performance' in config and 'sampling_frequency' in config['performance']:
        sampling_freq = config['performance']['sampling_frequency']
    
    # Calculate metrics with configurable sampling frequency
    stats = {
        "Strategy": {
            "Cumulative Return": (portvals.iloc[-1] / portvals.iloc[0]).values[0] - 1,
            "Average Daily Return": port_daily_rets.mean().values[0],
            "Std Dev Daily Return": port_daily_rets.std().values[0],
            "Sharpe Ratio": np.sqrt(sampling_freq) * port_daily_rets.mean().values[0] / port_daily_rets.std().values[0]
        },
        "Benchmark": {
            "Cumulative Return": (benchvals.iloc[-1] / benchvals.iloc[0]).values[0] - 1,
            "Average Daily Return": bench_daily_rets.mean().values[0],
            "Std Dev Daily Return": bench_daily_rets.std().values[0],
            "Sharpe Ratio": np.sqrt(sampling_freq) * bench_daily_rets.mean().values[0] / bench_daily_rets.std().values[0]
        }
    }
    
    return stats


def run_manual_strategy_test(symbol, sd, ed, sv=100000, commission=9.95, impact=0.005, config=None):
    """
    Run and evaluate manual strategy against benchmark.
    
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
    config : dict, optional
        Configuration dictionary
        
    Returns:
    --------
    tuple
        (trades, portfolio values, benchmark values, statistics)
    """
    # Get manual strategy parameters from config
    ms_params = {}
    position_size = 1000  # Default position size
    
    if config is not None:
        if 'manual_strategy' in config:
            # Copy all parameters from the manual_strategy section directly
            ms_params = config['manual_strategy'].copy()
            
        if 'portfolio' in config and 'max_positions' in config['portfolio']:
            # Override position_size if it wasn't in manual_strategy section
            if 'position_size' not in ms_params:
                position_size = config['portfolio']['max_positions']
                ms_params['position_size'] = position_size
    
    # Create strategy with parameters
    print_subsection("1/4 - Initializing ManualStrategy")
    ms = ManualStrategy(**ms_params)
    
    # Display the strategy parameters
    print_status(f"ManualStrategy initialized for {symbol} with parameters:")
    for param, value in ms_params.items():
        print(f"  - {param}: {value}")
    
    # Generate trades
    print_subsection("2/4 - Generating Trading Decisions")
    print_status(f"Generating Manual Strategy trades for {symbol}...")
    import time
    start_time = time.time()
    trades = ms.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    trading_time = time.time() - start_time
    trade_count = (trades != 0).sum()[0]
    print_status(f"Strategy generated {trade_count} trades in {trading_time:.2f} seconds")
    
    # Compute portfolio values
    print_subsection("3/4 - Simulating Portfolio Performance")
    print_status(f"Simulating Manual Strategy portfolio...")
    portvals = compute_portvals(
        orders=trades, 
        start_val=sv, 
        commission=commission, 
        impact=impact
    )
    
    # Create benchmark with configurable position size
    print_status(f"Creating benchmark (buy & hold) portfolio...")
    benchmark_trades = pd.DataFrame(index=trades.index)
    benchmark_trades[symbol] = 0
    benchmark_trades.iloc[0] = position_size  # Use configurable position size
    
    # Compute benchmark values
    benchvals = compute_portvals(
        orders=benchmark_trades, 
        start_val=sv, 
        commission=commission, 
        impact=impact
    )
    
    # Calculate performance stats
    print_subsection("4/4 - Calculating Performance Metrics")
    print_status(f"Computing performance statistics...")
    stats = calc_stats(portvals, benchvals, config=config)
    
    # Return results
    return trades, portvals, benchvals, stats


def run_evaluation_from_config(config_path):
    """
    Run strategy evaluation using parameters from a config file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML configuration file
    """
    # Load config
    config = load_config(config_path)
    
    # Extract parameters
    symbol = config['data']['symbol']
    initial_value = config['portfolio']['starting_value']
    
    # Get trading parameters
    trading_config = config.get('trading', {})
    commission = trading_config.get('commission', 9.95)
    impact = trading_config.get('impact', 0.005)
    
    # Get date ranges - handle flexible date format (support both direct and in/out-sample)
    if 'in_sample' in config['data']:
        # Use in/out sample format
        in_sample_start = dt.datetime.strptime(config['data']['in_sample']['start_date'], '%Y-%m-%d')
        in_sample_end = dt.datetime.strptime(config['data']['in_sample']['end_date'], '%Y-%m-%d')
        
        # Check for out-of-sample period
        run_out_of_sample = 'out_sample' in config['data']
        if run_out_of_sample:
            out_sample_start = dt.datetime.strptime(config['data']['out_sample']['start_date'], '%Y-%m-%d')
            out_sample_end = dt.datetime.strptime(config['data']['out_sample']['end_date'], '%Y-%m-%d')
    else:
        # Use direct date format
        in_sample_start = dt.datetime.strptime(config['data']['start_date'], '%Y-%m-%d')
        in_sample_end = dt.datetime.strptime(config['data']['end_date'], '%Y-%m-%d')
        run_out_of_sample = False
    
    # Get output file prefix
    output_prefix = config['experiment']['output_prefix']
    experiment_name = config['experiment'].get('name', 'Manual Strategy Evaluation')
    
    # Print experiment header with configuration summary
    print_section(f"MANUAL STRATEGY EVALUATION: {experiment_name}")
    
    print_status(f"Symbol: {symbol}")
    print_status(f"Portfolio: Starting value ${initial_value:,.2f}")
    print_status(f"Trading costs: Commission ${commission:.2f}, Market impact {impact:.3f}")
    
    # Show strategy configuration if available
    if 'manual_strategy' in config:
        print_subsection("Strategy Configuration")
        for param, value in config['manual_strategy'].items():
            print(f"  - {param}: {value}")
    
    # Run in-sample test with config
    print_section("IN-SAMPLE TEST")
    print_status(f"Date Range: {in_sample_start.strftime('%Y-%m-%d')} to {in_sample_end.strftime('%Y-%m-%d')}")
    trades_in, portvals_in, benchvals_in, stats_in = run_manual_strategy_test(
        symbol=symbol,
        sd=in_sample_start,
        ed=in_sample_end,
        sv=initial_value,
        commission=commission,
        impact=impact,
        config=config  # Pass config
    )
    
    # Plot in-sample results with config
    print_status(f"Creating performance chart...")
    plot_strategy_comparison(
        trades_in, 
        portvals_in, 
        benchvals_in,
        f"Manual Strategy vs Benchmark (In-Sample): {symbol}",
        f"{output_prefix}_in_sample.png",
        config=config  # Pass config
    )
    
    # Print in-sample statistics
    print_subsection("In-Sample Performance Summary")
    for metric in ["Cumulative Return", "Average Daily Return", "Std Dev Daily Return", "Sharpe Ratio"]:
        strat_value = stats_in['Strategy'][metric]
        bench_value = stats_in['Benchmark'][metric]
        # Colorize based on comparison
        if metric == "Sharpe Ratio" or metric == "Average Daily Return" or metric == "Cumulative Return":
            strat_color = Fore.GREEN if strat_value > bench_value else Fore.RED
            bench_color = Fore.GREEN if bench_value > strat_value else Fore.RED
        else:  # For volatility/std dev, lower is better
            strat_color = Fore.GREEN if strat_value < bench_value else Fore.RED
            bench_color = Fore.GREEN if bench_value < strat_value else Fore.RED
        
        print(f"{Fore.WHITE}{Style.BRIGHT}{metric}:{Style.RESET_ALL}")
        print(f"  Strategy: {strat_color}{strat_value:.6f}{Style.RESET_ALL}")
        print(f"  Benchmark: {bench_color}{bench_value:.6f}{Style.RESET_ALL}")
    
    # Run out-of-sample test if dates are provided
    if run_out_of_sample:
        print_section("OUT-OF-SAMPLE TEST")
        print_status(f"Date Range: {out_sample_start.strftime('%Y-%m-%d')} to {out_sample_end.strftime('%Y-%m-%d')}")
        trades_out, portvals_out, benchvals_out, stats_out = run_manual_strategy_test(
            symbol=symbol,
            sd=out_sample_start,
            ed=out_sample_end,
            sv=initial_value,
            commission=commission,
            impact=impact,
            config=config  # Pass config
        )
        
        # Plot out-of-sample results with config
        print_status(f"Creating performance chart...")
        plot_strategy_comparison(
            trades_out, 
            portvals_out, 
            benchvals_out,
            f"Manual Strategy vs Benchmark (Out-of-Sample): {symbol}",
            f"{output_prefix}_out_sample.png",
            config=config  # Pass config
        )
        
        # Print out-of-sample statistics
        print_subsection("Out-of-Sample Performance Summary")
        for metric in ["Cumulative Return", "Average Daily Return", "Std Dev Daily Return", "Sharpe Ratio"]:
            strat_value = stats_out['Strategy'][metric]
            bench_value = stats_out['Benchmark'][metric]
            # Colorize based on comparison
            if metric == "Sharpe Ratio" or metric == "Average Daily Return" or metric == "Cumulative Return":
                strat_color = Fore.GREEN if strat_value > bench_value else Fore.RED
                bench_color = Fore.GREEN if bench_value > strat_value else Fore.RED
            else:  # For volatility/std dev, lower is better
                strat_color = Fore.GREEN if strat_value < bench_value else Fore.RED
                bench_color = Fore.GREEN if bench_value < strat_value else Fore.RED
            
            print(f"{Fore.WHITE}{Style.BRIGHT}{metric}:{Style.RESET_ALL}")
            print(f"  Strategy: {strat_color}{strat_value:.6f}{Style.RESET_ALL}")
            print(f"  Benchmark: {bench_color}{bench_value:.6f}{Style.RESET_ALL}")
    
    # Print summary of outputs
    print_section("EXPERIMENT OUTPUTS")
    print(f"  - In-sample chart: {os.path.join(get_output_path(), f'{output_prefix}_in_sample.png')}")
    if run_out_of_sample:
        print(f"  - Out-of-sample chart: {os.path.join(get_output_path(), f'{output_prefix}_out_sample.png')}")
    print(f"\n{Fore.GREEN}{Style.BRIGHT}Evaluation completed successfully!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "="*80 + f"{Style.RESET_ALL}")


def main():
    """Main function to run manual strategy evaluation from config file."""
    parser = argparse.ArgumentParser(
        description='Evaluate manual strategy against market benchmark'
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Run evaluation with config
    run_evaluation_from_config(args.config)


if __name__ == "__main__":
    main()