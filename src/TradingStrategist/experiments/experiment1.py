"""
Experiment 1: Comparison of Manual vs. Tree-based ML Strategy

This experiment compares the performance of a rule-based manual strategy 
against a decision tree-based machine learning strategy.
"""

import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import yaml
import time
from colorama import init, Fore, Style

# Initialize colorama for colored terminal output
init()

from TradingStrategist.models.ManualStrategy import ManualStrategy
from TradingStrategist.models.TreeStrategyLearner import TreeStrategyLearner
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


def run_experiment(config_path):
    """
    Run experiment comparing manual and ML strategies.
    
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
    
    # Get experiment name
    experiment_name = experiment_config.get('name', 'Experiment 1')
    output_prefix = experiment_config.get('output_prefix', 'exp1')
    
    # Get data parameters
    symbol = data_config.get('symbol', 'JPM')
    train_start = dt.datetime.strptime(data_config['in_sample']['start_date'], '%Y-%m-%d')
    train_end = dt.datetime.strptime(data_config['in_sample']['end_date'], '%Y-%m-%d')
    test_start = dt.datetime.strptime(data_config['out_sample']['start_date'], '%Y-%m-%d')
    test_end = dt.datetime.strptime(data_config['out_sample']['end_date'], '%Y-%m-%d')
    
    # Get portfolio parameters
    starting_value = portfolio_config.get('starting_value', 100000)
    
    # Get trading parameters
    commission = trading_config.get('commission', 9.95)
    impact = trading_config.get('impact', 0.005)
    
    # Print experiment setup
    print_section(f"EXPERIMENT 1: {experiment_name}")
    
    print_status(f"Comparison: Manual Strategy vs. TreeStrategy ML Approach")
    print_status(f"Symbol: {symbol}")
    
    print_subsection("Time Periods")
    print(f"  Training: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
    print(f"  Testing:  {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
    
    print_subsection("Portfolio Settings")
    print(f"  - Starting value: ${starting_value:,.2f}")
    print(f"  - Commission: ${commission:.2f}")
    print(f"  - Market impact: {impact:.3f}")
    
    # Create and train the TreeStrategyLearner
    print_subsection("1/4 - Training TreeStrategyLearner Model")
    learner = TreeStrategyLearner(
        verbose=True,
        impact=impact,
        commission=commission,
        **dt_config
    )
    
    print(f"Using TreeStrategyLearner with parameters:")
    for key, value in dt_config.items():
        print(f"  - {key}: {value}")
    
    print_status(f"Training model for {symbol} from {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}...")
    start_time = time.time()
    learner.addEvidence(
        symbol=symbol,
        sd=train_start,
        ed=train_end,
        sv=starting_value
    )
    training_time = time.time() - start_time
    print_status(f"Model training completed in {training_time:.2f} seconds")
    
    # Create manual strategy instance
    print_subsection("2/4 - Initializing ManualStrategy")
    manual_strategy = ManualStrategy(
        position_size=dt_config.get('position_size', 1000),
        window_size=dt_config.get('window_size', 20),
        rsi_window=dt_config.get('rsi_window', 14),
        stoch_window=dt_config.get('stoch_window', 14),
        cci_window=dt_config.get('cci_window', 20),
        buy_threshold=dt_config.get('buy_threshold', 0.02),
        sell_threshold=dt_config.get('sell_threshold', -0.02),
        verbose=True
    )
    
    print_status(f"ManualStrategy initialized with trading parameters:")
    for param in ['position_size', 'window_size', 'rsi_window', 'stoch_window', 'cci_window', 'buy_threshold', 'sell_threshold']:
        if param == 'position_size':
            # position_size is a direct attribute
            print(f"  - {param}: {manual_strategy.position_size}")
        elif param in ['buy_threshold', 'sell_threshold']:
            # These may not be in thresholds dictionary as they're converted to voting thresholds
            if param in manual_strategy.thresholds:
                print(f"  - {param}: {manual_strategy.thresholds[param]}")
            else:
                # If not directly available, they were converted to voting thresholds
                voting_param = 'min_vote_buy' if param == 'buy_threshold' else 'min_vote_sell'
                print(f"  - {param}: using {voting_param}={manual_strategy.thresholds[voting_param]}")
        else:
            # Other parameters are in the thresholds dictionary
            print(f"  - {param}: {manual_strategy.thresholds[param]}")
    
    # Generate trades for the test period
    print_subsection("3/4 - Generating Trading Decisions")
    
    # ML Strategy
    print_status(f"Generating ML Strategy trades for {symbol}...")
    start_time = time.time()
    ml_trades = learner.testPolicy(
        symbol=symbol,
        sd=test_start,
        ed=test_end,
        sv=starting_value
    )
    ml_time = time.time() - start_time
    ml_trade_count = (ml_trades != 0).sum()[0]
    print_status(f"ML Strategy generated {ml_trade_count} trades in {ml_time:.2f} seconds")
    
    # Manual Strategy
    print_status(f"Generating Manual Strategy trades for {symbol}...")
    start_time = time.time()
    manual_trades = manual_strategy.testPolicy(
        symbol=symbol,
        sd=test_start,
        ed=test_end,
        sv=starting_value
    )
    manual_time = time.time() - start_time
    manual_trade_count = (manual_trades != 0).sum()[0]
    print_status(f"Manual Strategy generated {manual_trade_count} trades in {manual_time:.2f} seconds")
    
    # Benchmark - Buy and Hold
    print_status(f"Generating Benchmark (buy & hold) trades...")
    benchmark_trades = pd.DataFrame(0, index=ml_trades.index, columns=[symbol])
    benchmark_trades.iloc[0] = dt_config.get('position_size', 1000)  # Buy on first day
    
    # Run market simulator
    print_subsection("4/4 - Simulating Portfolio Performance")
    
    # ML Strategy portfolio
    print_status(f"Simulating ML Strategy portfolio...")
    ml_portvals = compute_portvals(
        orders=ml_trades,
        start_val=starting_value,
        commission=commission,
        impact=impact
    )
    
    # Manual Strategy portfolio
    print_status(f"Simulating Manual Strategy portfolio...")
    manual_portvals = compute_portvals(
        orders=manual_trades,
        start_val=starting_value,
        commission=commission,
        impact=impact
    )
    
    # Benchmark portfolio
    print_status(f"Simulating Benchmark portfolio...")
    benchmark_portvals = compute_portvals(
        orders=benchmark_trades,
        start_val=starting_value,
        commission=commission,
        impact=impact
    )
    
    # Calculate statistics
    print_subsection("Calculating Performance Metrics")
    
    stats_df = calculate_performance_stats(
        ml_portvals=ml_portvals,
        manual_portvals=manual_portvals,
        benchmark_portvals=benchmark_portvals,
        ml_trades=ml_trades,
        manual_trades=manual_trades
    )
    
    # Generate plots and save results
    output_path = get_output_path()
    date_str = test_start.strftime('%Y%m%d') + '_' + test_end.strftime('%Y%m%d')
    
    # Filenames for outputs
    plot_filename = f"{output_prefix}_{symbol}_comparison_{date_str}.png"
    stats_filename = f"{output_prefix}_{symbol}_stats_{date_str}.csv"
    feature_imp_filename = f"{output_prefix}_{symbol}_feature_importance_{date_str}.png"
    
    # Plot comparison
    print_subsection("Generating Charts and Saving Results")
    print_status(f"Creating strategy comparison chart...")
    plot_comparison(
        ml_portvals=ml_portvals,
        manual_portvals=manual_portvals,
        benchmark_portvals=benchmark_portvals,
        ml_trades=ml_trades,
        manual_trades=manual_trades,
        symbol=symbol,
        title=f"Strategy Comparison: Manual vs. ML ({symbol})",
        output_path=os.path.join(output_path, plot_filename)
    )
    print_file_saved(os.path.join(output_path, plot_filename))
    
    # Save statistics
    print_status(f"Saving performance statistics...")
    stats_df.to_csv(os.path.join(output_path, stats_filename))
    print_file_saved(os.path.join(output_path, stats_filename))
    
    # Plot feature importance if available
    if hasattr(learner, 'get_feature_importances'):
        print_status(f"Generating feature importance visualization...")
        feature_importance = learner.get_feature_importances()
        if feature_importance is not None:
            plot_feature_importance(
                feature_importance,
                title=f"Feature Importance: Tree Strategy ({symbol})",
                output_path=os.path.join(output_path, feature_imp_filename)
            )
            print_file_saved(os.path.join(output_path, feature_imp_filename))
    
    # Print statistics
    print_subsection("PERFORMANCE STATISTICS")
    print(f"\n{stats_df}")
    
    # Print summary of trades
    print_subsection("TRADE SUMMARY")
    print(f"Manual Strategy: {manual_trade_count} trades")
    print(f"ML Strategy:     {ml_trade_count} trades")
    print(f"Benchmark:       1 trade (initial buy)")
    
    print_subsection("EXPERIMENT OUTPUTS")
    print(f"  - Comparison chart: {os.path.join(output_path, plot_filename)}")
    print(f"  - Statistics file:  {os.path.join(output_path, stats_filename)}")
    if hasattr(learner, 'get_feature_importances') and feature_importance is not None:
        print(f"  - Feature importance: {os.path.join(output_path, feature_imp_filename)}")
    print(f"\n{Fore.GREEN}{Style.BRIGHT}Experiment completed successfully!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "="*80 + f"{Style.RESET_ALL}")


def calculate_performance_stats(ml_portvals, manual_portvals, benchmark_portvals, 
                               ml_trades, manual_trades):
    """
    Calculate performance statistics for the strategies.
    
    Parameters:
    -----------
    ml_portvals : pd.DataFrame
        Portfolio values for ML strategy
    manual_portvals : pd.DataFrame
        Portfolio values for Manual strategy
    benchmark_portvals : pd.DataFrame
        Portfolio values for benchmark
    ml_trades : pd.DataFrame
        Trades for ML strategy
    manual_trades : pd.DataFrame
        Trades for Manual strategy
        
    Returns:
    --------
    pd.DataFrame
        Performance statistics
    """
    # Calculate daily returns
    ml_returns = ml_portvals.pct_change().dropna()
    manual_returns = manual_portvals.pct_change().dropna()
    benchmark_returns = benchmark_portvals.pct_change().dropna()
    
    # Calculate metrics for each strategy
    ml_cum_ret = (ml_portvals.iloc[-1] / ml_portvals.iloc[0]) - 1
    manual_cum_ret = (manual_portvals.iloc[-1] / manual_portvals.iloc[0]) - 1
    benchmark_cum_ret = (benchmark_portvals.iloc[-1] / benchmark_portvals.iloc[0]) - 1
    
    ml_avg_ret = ml_returns.mean()
    manual_avg_ret = manual_returns.mean()
    benchmark_avg_ret = benchmark_returns.mean()
    
    ml_std_ret = ml_returns.std()
    manual_std_ret = manual_returns.std()
    benchmark_std_ret = benchmark_returns.std()
    
    ml_sharpe = np.sqrt(252) * ml_avg_ret / ml_std_ret
    manual_sharpe = np.sqrt(252) * manual_avg_ret / manual_std_ret
    benchmark_sharpe = np.sqrt(252) * benchmark_avg_ret / benchmark_std_ret
    
    # Count trades
    ml_trade_count = (ml_trades != 0).sum()[0]
    manual_trade_count = (manual_trades != 0).sum()[0]
    
    # Create DataFrame
    stats_df = pd.DataFrame({
        "Cumulative Return": [manual_cum_ret.values[0], ml_cum_ret.values[0], benchmark_cum_ret.values[0]],
        "Average Daily Return": [manual_avg_ret.values[0], ml_avg_ret.values[0], benchmark_avg_ret.values[0]],
        "Std Dev Daily Return": [manual_std_ret.values[0], ml_std_ret.values[0], benchmark_std_ret.values[0]],
        "Sharpe Ratio": [manual_sharpe.values[0], ml_sharpe.values[0], benchmark_sharpe.values[0]],
        "Trade Count": [manual_trade_count, ml_trade_count, 1]
    }, index=["Manual Strategy", "ML Strategy", "Benchmark"])
    
    return stats_df


def plot_comparison(ml_portvals, manual_portvals, benchmark_portvals, ml_trades, 
                   manual_trades, symbol, title, output_path=None):
    """
    Plot performance comparison of strategies.
    
    Parameters:
    -----------
    ml_portvals : pd.DataFrame
        Portfolio values for ML strategy
    manual_portvals : pd.DataFrame
        Portfolio values for Manual strategy
    benchmark_portvals : pd.DataFrame
        Portfolio values for benchmark
    ml_trades : pd.DataFrame
        Trades for ML strategy (for visualization)
    manual_trades : pd.DataFrame
        Trades for Manual strategy (for visualization)
    symbol : str
        Stock symbol
    title : str
        Plot title
    output_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Normalize values
    norm_ml = ml_portvals / ml_portvals.iloc[0]
    norm_manual = manual_portvals / manual_portvals.iloc[0]
    norm_benchmark = benchmark_portvals / benchmark_portvals.iloc[0]
    
    # Plot portfolios
    plt.plot(norm_ml, label="ML Strategy", color="blue", linewidth=2)
    plt.plot(norm_manual, label="Manual Strategy", color="red", linewidth=1.5)
    plt.plot(norm_benchmark, label="Benchmark (Buy & Hold)", color="green", linestyle="--")
    
    # Add shaded regions for trades (optional visualization)
    for date, val in manual_trades.itertuples():
        if val > 0:
            plt.axvline(date, color="red", linestyle="--", alpha=0.2)
        elif val < 0:
            plt.axvline(date, color="black", linestyle="--", alpha=0.2)
    
    # Annotate plot
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Portfolio Value")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    
    # Add text with performance info
    ml_return = (norm_ml.iloc[-1].values[0] - 1) * 100
    manual_return = (norm_manual.iloc[-1].values[0] - 1) * 100
    benchmark_return = (norm_benchmark.iloc[-1].values[0] - 1) * 100
    
    plt.annotate(f"ML Return: {ml_return:.1f}%", 
                xy=(0.02, 0.95), xycoords="axes fraction", fontsize=10)
    plt.annotate(f"Manual Return: {manual_return:.1f}%", 
                xy=(0.02, 0.91), xycoords="axes fraction", fontsize=10)
    plt.annotate(f"Benchmark Return: {benchmark_return:.1f}%", 
                xy=(0.02, 0.87), xycoords="axes fraction", fontsize=10)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_feature_importance(feature_importance, title, output_path=None):
    """
    Plot feature importance from ML model.
    
    Parameters:
    -----------
    feature_importance : dict
        Dictionary of feature names and importance values
    title : str
        Plot title
    output_path : str, optional
        Path to save the plot
    """
    if not feature_importance:
        return
        
    plt.figure(figsize=(10, 6))
    
    # Sort features by importance
    items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in items]
    importance = [item[1] for item in items]
    
    # Create bar chart
    bars = plt.bar(features, importance, color='steelblue')
    
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height * 1.01,
            f"{height:.3f}",
            ha='center', 
            va='bottom', 
            fontsize=9
        )
    
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main():
    """Run the experiment."""
    parser = argparse.ArgumentParser(description='Run Experiment 1: Manual vs. ML Strategy')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    run_experiment(args.config)


if __name__ == "__main__":
    main()