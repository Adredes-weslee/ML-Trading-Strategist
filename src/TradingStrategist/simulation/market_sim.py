"""
Market Simulation Module

This module provides market simulation functionality for evaluating trading strategies.
It includes portfolio value calculation and performance metrics assessment.
"""

import pandas as pd
import numpy as np
from TradingStrategist.data.loader import get_data


def compute_portvals(orders_df, start_val=100000, commission=9.95, impact=0.005, config=None):
    """
    Compute portfolio values based on trade orders.
    
    Parameters:
    -----------
    orders_df : pd.DataFrame
        Trade orders dataframe with date index and columns for each symbol
        Values should be number of shares to buy (positive) or sell (negative)
    start_val : int, optional
        Starting portfolio value, default 100000
    commission : float, optional
        Trading commission per trade, default 9.95
    impact : float, optional
        Market impact per trade as a percentage of trade value, default 0.005 (0.5%)
    config : dict, optional
        Configuration dictionary with simulation parameters
        
    Returns:
    --------
    pd.DataFrame
        Portfolio value over time
    """
    # Use config if provided, otherwise use default values
    if config is not None:
        if 'trading' in config:
            commission = config['trading'].get('commission', commission)
            impact = config['trading'].get('impact', impact)
        if 'portfolio' in config:
            start_val = config['portfolio'].get('starting_value', start_val)
    
    if orders_df.empty:
        return pd.DataFrame({'portfolio_value': [start_val]}, index=[orders_df.index[0]])

    # Get list of symbols
    symbols = orders_df.columns.tolist()
    
    # Get price data for all symbols
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates, addSPY=False)
    
    # Forward fill any missing prices
    prices_all = prices_all.fillna(method='ffill')
    prices_all = prices_all.fillna(method='bfill')
    
    # Create trades dataframe
    trades = pd.DataFrame(0.0, index=prices_all.index, columns=symbols)
    
    # Track cash and holdings
    holdings = pd.DataFrame(0, index=prices_all.index, columns=symbols)
    cash = pd.Series(start_val, index=prices_all.index)

    # Process orders chronologically
    for date in orders_df.index:
        for symbol in symbols:
            shares = orders_df.loc[date, symbol]
            if shares == 0:
                continue
            
            price = prices_all.loc[date, symbol]
            
            # Calculate cost of trade including impact
            impact_factor = 1.0 + (impact * np.sign(shares))
            trade_cost = abs(shares) * price * impact_factor
            
            # Update holdings and cash
            trades.loc[date, symbol] = shares
            cash[date] -= trade_cost + commission if shares != 0 else 0
    
    # Calculate cumulative holdings
    for symbol in symbols:
        holdings[symbol] = trades[symbol].cumsum()
    
    # Propagate cash forward
    cash = cash.cumsum()
    
    # Calculate daily portfolio value
    values = pd.DataFrame(index=prices_all.index, columns=['portfolio_value'])
    
    for date in prices_all.index:
        # Stock values
        stock_values = 0
        for symbol in symbols:
            stock_values += holdings.loc[date, symbol] * prices_all.loc[date, symbol]
        
        # Total value is stock values plus cash
        values.loc[date, 'portfolio_value'] = stock_values + cash[date]
    
    return values


def assess_portfolio(portfolio_values, rfr=0.0, sf=252.0, config=None):
    """
    Calculate portfolio performance metrics.
    
    Parameters:
    -----------
    portfolio_values : pd.DataFrame
        Portfolio value over time
    rfr : float, optional
        Risk-free rate (annualized), default 0.0
    sf : float, optional
        Sampling frequency (e.g., 252 trading days per year), default 252.0
    config : dict, optional
        Configuration dictionary with assessment parameters
        
    Returns:
    --------
    dict
        Dictionary of performance metrics
    """
    # Use config if provided, otherwise use default values
    if config is not None and 'performance' in config:
        rfr = config['performance'].get('risk_free_rate', rfr)
        sf = config['performance'].get('sampling_frequency', sf)
    
    # Extract portfolio value series
    if isinstance(portfolio_values, pd.DataFrame):
        portvals = portfolio_values['portfolio_value']
    else:
        portvals = portfolio_values
    
    # Calculate daily returns
    daily_returns = portvals.pct_change().dropna()
    
    # Calculate metrics
    cumulative_return = (portvals.iloc[-1] / portvals.iloc[0]) - 1.0
    avg_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    
    # Annualize returns and calculate Sharpe ratio
    annualized_return = (1.0 + avg_daily_return) ** sf - 1.0
    annualized_risk = std_daily_return * np.sqrt(sf)
    sharpe_ratio = np.sqrt(sf) * (avg_daily_return - rfr/sf) / std_daily_return if std_daily_return > 0 else 0
    
    # Calculate maximum drawdown
    peak = portvals.expanding().max()
    drawdown = (portvals / peak) - 1.0
    max_drawdown = drawdown.min()
    
    return {
        'cumulative_return': cumulative_return,
        'avg_daily_return': avg_daily_return,
        'std_daily_return': std_daily_return,
        'sharpe_ratio': sharpe_ratio,
        'annualized_return': annualized_return,
        'annualized_risk': annualized_risk,
        'max_drawdown': max_drawdown
    }