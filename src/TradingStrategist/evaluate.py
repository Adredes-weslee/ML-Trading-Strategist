"""
Evaluation Module

This module provides functions for evaluating trading strategies,
including benchmark comparison and performance metrics calculation.
"""

import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from TradingStrategist.models.ManualStrategy import ManualStrategy  # Fixed CamelCase
from TradingStrategist.models.TreeStrategyLearner import TreeStrategyLearner  # Fixed class name
from TradingStrategist.models.QStrategyLearner import QStrategyLearner
from TradingStrategist.data.loader import get_data
from TradingStrategist.simulation.market_sim import compute_portvals
from TradingStrategist.utils.helpers import get_output_path, load_config


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
        'cumulative_return': (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1,
        'average_daily_return': daily_returns.mean(),
        'std_daily_return': daily_returns.std(),
        'sharpe_ratio': np.sqrt(252) * daily_returns.mean() / daily_returns.std(),
        'final_value': portfolio_values.iloc[-1]
    }
    
    return metrics


def plot_performance_comparison(results, title, filename, output_path):
    """
    Plot performance comparison of different strategies.
    
    Parameters:
    -----------
    results : dict
        Dictionary mapping strategy names to portfolio values
    title : str
        Plot title
    filename : str
        Output filename
    output_path : str
        Path to save output files
    """
    plt.figure(figsize=(10, 6))
    
    colors = {
        'Benchmark': 'black',
        'Manual Strategy': 'red',
        'Strategy Learner': 'blue',
        'Q-Strategy Learner': 'green'
    }
    
    for strategy_name, portfolio_values in results.items():
        # Normalize values
        norm_values = portfolio_values / portfolio_values.iloc[0]
        plt.plot(norm_values, label=strategy_name, color=colors.get(strategy_name, 'purple'))
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Normalized Portfolio Value')
    plt.grid(True)
    plt.legend(loc='best')
    
    # Save figure
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path)
    plt.close()
    
    return save_path


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
    benchmark_values = compute_portvals(orders_df=trades, 
                                       start_val=starting_value, 
                                       commission=commission, 
                                       impact=impact)
    
    return benchmark_values


def evaluate_model(model_name, symbol, start_date, end_date, 
                   training_start=None, training_end=None,
                   starting_value=100000, commission=9.95, impact=0.005,
                   strategy_params=None, q_strategy_params=None):
    """
    Evaluate a trading model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model to evaluate ('manual', 'strategy_learner', 'q_strategy_learner')
    symbol : str
        Stock symbol to trade
    start_date : datetime
        Start date for evaluation
    end_date : datetime
        End date for evaluation
    training_start : datetime, optional
        Start date for training (ML strategies only)
    training_end : datetime, optional
        End date for training (ML strategies only)
    starting_value : int, optional
        Starting portfolio value
    commission : float, optional
        Commission per trade
    impact : float, optional
        Market impact per trade
    strategy_params : dict, optional
        Parameters for TreeStrategyLearner
    q_strategy_params : dict, optional
        Parameters for QStrategyLearner
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing portfolio values
    """
    if model_name == "manual":
        model = ManualStrategy(verbose=False)
    elif model_name == "strategy_learner":
        if strategy_params is None:
            strategy_params = {}
        
        model = TreeStrategyLearner(  # Fixed class name
            verbose=False, 
            impact=impact,
            commission=commission,
            window_size=strategy_params.get("window_size", 20),
            buy_threshold=strategy_params.get("buy_threshold", 0.02),
            sell_threshold=strategy_params.get("sell_threshold", -0.02),
            prediction_days=strategy_params.get("prediction_days", 5),
            leaf_size=strategy_params.get("leaf_size", 5),
            bags=strategy_params.get("bags", 20),
            boost=strategy_params.get("boost", False)
        )
    elif model_name == "q_strategy_learner":
        if q_strategy_params is None:
            q_strategy_params = {}
        
        model = QStrategyLearner(
            verbose=False,
            impact=impact,
            commission=commission,
            indicator_bins=q_strategy_params.get("indicator_bins", 10),
            window_size=q_strategy_params.get("window_size", 20),
            rsi_window=q_strategy_params.get("rsi_window", 14),
            position_size=q_strategy_params.get("position_size", 1000),
            max_iterations=q_strategy_params.get("max_iterations", 100),
            learning_rate=q_strategy_params.get("learning_rate", 0.2),
            discount_factor=q_strategy_params.get("discount_factor", 0.9),
            random_action_rate=q_strategy_params.get("random_action_rate", 0.5),
            random_action_decay=q_strategy_params.get("random_action_decay", 0.99),
            dyna_iterations=q_strategy_params.get("dyna_iterations", 10)
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Train if needed (for ML strategies)
    if model_name in ["strategy_learner", "q_strategy_learner"]:
        # Use training period if provided, otherwise use evaluation period
        train_start = training_start if training_start is not None else start_date
        train_end = training_end if training_end is not None else end_date
        
        print(f"Training {model_name} from {train_start} to {train_end}")
        model.addEvidence(symbol=symbol, sd=train_start, ed=train_end, sv=starting_value)
    
    # Generate trades
    print(f"Evaluating {model_name} from {start_date} to {end_date}")
    trades = model.testPolicy(symbol=symbol, sd=start_date, ed=end_date, sv=starting_value)
    
    # Count trades
    trade_count = (trades != 0).sum().sum()
    print(f"  Generated {trade_count} trades")
    
    # Compute portfolio values
    portvals = compute_portvals(
        orders_df=trades, 
        start_val=starting_value, 
        commission=commission, 
        impact=impact
    )
    
    return portvals


def run_evaluation(config_path):
    """
    Run evaluation based on config file.
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    """
    # Load config
    config = load_config(config_path)
    
    # Get experiment details
    experiment_name = config['experiment'].get('name', 'Strategy Evaluation')
    output_prefix = config['experiment'].get('output_prefix', 'evaluation')
    
    # Extract data parameters
    symbol = config['data']['symbol']
    
    # Get date ranges
    if 'training' in config['data'] and 'testing' in config['data']:
        train_start = dt.datetime.strptime(config['data']['training']['start_date'], '%Y-%m-%d')
        train_end = dt.datetime.strptime(config['data']['training']['end_date'], '%Y-%m-%d')
        test_start = dt.datetime.strptime(config['data']['testing']['start_date'], '%Y-%m-%d')
        test_end = dt.datetime.strptime(config['data']['testing']['end_date'], '%Y-%m-%d')
    elif 'in_sample' in config['data'] and 'out_sample' in config['data']:
        train_start = dt.datetime.strptime(config['data']['in_sample']['start_date'], '%Y-%m-%d')
        train_end = dt.datetime.strptime(config['data']['in_sample']['end_date'], '%Y-%m-%d')
        test_start = dt.datetime.strptime(config['data']['out_sample']['start_date'], '%Y-%m-%d')
        test_end = dt.datetime.strptime(config['data']['out_sample']['end_date'], '%Y-%m-%d')
    else:
        # Just one date range
        single_start = dt.datetime.strptime(config['data']['start_date'], '%Y-%m-%d')
        single_end = dt.datetime.strptime(config['data']['end_date'], '%Y-%m-%d')
        train_start = test_start = single_start
        train_end = test_end = single_end
    
    # Get portfolio parameters
    starting_value = config['portfolio']['starting_value']
    
    # Get trading parameters
    trading_config = config.get('trading', {})
    commission = trading_config.get('commission', 9.95)
    impact = trading_config.get('impact', 0.005)
    
    # Get strategy parameters - check both naming conventions
    if 'dt_strategy' in config:
        strategy_params = config['dt_strategy']
    elif 'strategy_learner' in config:
        strategy_params = config['strategy_learner']
    else:
        strategy_params = {}
    
    # Get Q-strategy parameters
    q_strategy_params = config.get('q_strategy', {})
    
    # Get visualization parameters
    viz_config = config.get('visualization', {})
    save_figures = viz_config.get('save_figures', True)
    compare_with_benchmark = viz_config.get('compare_with_benchmark', True)
    include_manual = viz_config.get('compare_with_manual', False)
    include_strategy_learner = viz_config.get('compare_with_strategy_learner', False)
    include_q_strategy = viz_config.get('compare_with_q_strategy', True)
    
    # Determine which models to evaluate
    models_to_evaluate = []
    
    if 'models' in config:
        # Explicit list of models
        if 'manual' in config['models']:
            models_to_evaluate.append('manual')
        if 'strategy_learner' in config['models']:
            models_to_evaluate.append('strategy_learner')
        if 'q_strategy_learner' in config['models']:
            models_to_evaluate.append('q_strategy_learner')
    else:
        # Implicit based on visualization config
        if include_manual:
            models_to_evaluate.append('manual')
        if include_strategy_learner:
            models_to_evaluate.append('strategy_learner')
        if include_q_strategy:
            models_to_evaluate.append('q_strategy_learner')
    
    # Create benchmark if needed
    results = {}
    if compare_with_benchmark:
        print("Creating benchmark...")
        benchmark_values = create_benchmark(
            symbol=symbol,
            start_date=test_start,
            end_date=test_end,
            starting_value=starting_value,
            commission=commission,
            impact=impact
        )
        results['Benchmark'] = benchmark_values
    
    # Evaluate models
    for model_name in models_to_evaluate:
        print(f"\nEvaluating {model_name}...")
        
        if model_name == "strategy_learner":
            portvals = evaluate_model(
                model_name, symbol, test_start, test_end,
                training_start=train_start, training_end=train_end,
                starting_value=starting_value, commission=commission, impact=impact,
                strategy_params=strategy_params
            )
            results['Strategy Learner'] = portvals
        elif model_name == "q_strategy_learner":
            portvals = evaluate_model(
                model_name, symbol, test_start, test_end,
                training_start=train_start, training_end=train_end,
                starting_value=starting_value, commission=commission, impact=impact,
                q_strategy_params=q_strategy_params
            )
            results['Q-Strategy Learner'] = portvals
        else:
            # Manual strategy
            portvals = evaluate_model(
                model_name, symbol, test_start, test_end,
                starting_value=starting_value, commission=commission, impact=impact
            )
            results['Manual Strategy'] = portvals
    
    # Calculate and display metrics
    print("\n" + "="*80)
    print(f"Performance Metrics ({symbol}):")
    print("="*80)
    
    metrics_list = []
    for strategy_name, portfolio_values in results.items():
        metrics = calculate_metrics(portfolio_values)
        metrics_row = {
            'Strategy': strategy_name,
            'Cumulative Return': f"{metrics['cumulative_return'].values[0]:.4f}",
            'Average Daily Return': f"{metrics['average_daily_return'].values[0]:.6f}",
            'Std Dev Daily Return': f"{metrics['std_daily_return'].values[0]:.6f}",
            'Sharpe Ratio': f"{metrics['sharpe_ratio'].values[0]:.4f}",
            'Final Value': f"${metrics['final_value'].values[0]:.2f}"
        }
        metrics_list.append(metrics_row)
    
    # Convert to DataFrame for pretty printing
    metrics_df = pd.DataFrame(metrics_list).set_index('Strategy')
    print(metrics_df)
    
    # Create output directory if needed
    output_path = get_output_path()
    os.makedirs(output_path, exist_ok=True)
    
    # Save metrics to CSV
    metrics_file = os.path.join(output_path, f"{output_prefix}_metrics.csv")
    metrics_df.to_csv(metrics_file)
    print(f"\nMetrics saved to: {metrics_file}")
    
    # Plot performance comparison
    if save_figures and len(results) > 0:
        title = f"{experiment_name}: {symbol} ({test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')})"
        plot_file = f"{output_prefix}_performance.png"
        
        save_path = plot_performance_comparison(
            results=results,
            title=title,
            filename=plot_file,
            output_path=output_path
        )
        print(f"Performance plot saved to: {save_path}")
    
    return results, metrics_df


def main():
    """
    Main function to run evaluation from command line.
    """
    parser = argparse.ArgumentParser(description='Evaluate trading strategies')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Run evaluation with config
    run_evaluation(args.config)


if __name__ == "__main__":
    main()