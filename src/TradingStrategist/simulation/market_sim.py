"""
Market Simulator

This module provides functionality for simulating market trading
and evaluating trading strategy performance.
"""

import pandas as pd
import numpy as np


def compute_portvals(orders, start_val=100000.0, commission=9.95, impact=0.005,
                     start_date=None, end_date=None, prices_df=None):
    """
    Compute portfolio values over time based on a set of orders.
    
    Parameters:
    -----------
    orders : pd.DataFrame
        Orders DataFrame with dates as index and symbols as columns, 
        values represent shares to trade (positive for buy, negative for sell)
    start_val : float, optional
        Starting portfolio value, default 100000
    commission : float, optional
        Fixed commission for each trade, default 9.95
    impact : float, optional
        Market impact per share traded, default 0.005
    start_date : datetime, optional
        Start date for portfolio simulation
    end_date : datetime, optional
        End date for portfolio simulation
    prices_df : pd.DataFrame, optional
        Prices DataFrame if already available
        
    Returns:
    --------
    pd.DataFrame
        Portfolio values over time
    """
    from TradingStrategist.data.loader import get_data
    
    # Get the unique symbols from the orders DataFrame
    symbols = orders.columns.tolist()
    
    # If start and end dates are not provided, infer from orders
    if start_date is None:
        start_date = orders.index.min()
    if end_date is None:
        end_date = orders.index.max()
    
    # Get price data if not provided
    if prices_df is None:
        prices_all = get_data(symbols, pd.date_range(start_date, end_date))
        
        # Remove SPY if it was added automatically
        prices_df = prices_all[symbols]
    
    # Forward-fill and backward-fill prices to handle missing data
    prices_all = prices_df.copy()
    prices_all = prices_all.ffill()
    prices_all = prices_all.bfill()
    
    # Initialize holdings and values DataFrames with float64 dtype to avoid dtype warnings
    holdings = pd.DataFrame(0.0, index=prices_all.index, columns=symbols, dtype=np.float64)
    values = pd.DataFrame(0.0, index=prices_all.index, columns=symbols, dtype=np.float64)
    
    # Initialize cash Series with float64 dtype
    cash = pd.Series(float(start_val), index=prices_all.index, dtype=np.float64)
    
    # Process orders and update holdings and cash
    for date in orders.index:
        if date not in prices_all.index:
            continue
        
        for symbol in symbols:
            shares = orders.loc[date, symbol]
            if shares == 0:
                continue
            
            # Calculate transaction cost (commission + impact)
            price = prices_all.loc[date, symbol]
            trade_cost = abs(shares) * price * impact
            commission_cost = commission if shares != 0 else 0
            
            # Update holdings
            holdings.loc[date:, symbol] += float(shares)
            
            # Update cash - already float dtype
            cash_value = cash[date] - (shares * price + trade_cost + commission_cost)
            cash.loc[date:] = cash_value
    
    # Calculate daily value of each position
    for date in prices_all.index:
        for symbol in symbols:
            values.loc[date, symbol] = holdings.loc[date, symbol] * prices_all.loc[date, symbol]
    
    # Calculate portfolio value (cash + sum of all position values)
    portval = pd.DataFrame(index=prices_all.index)
    portval['portfolio_value'] = cash + values.sum(axis=1)
    
    return portval


def assess_portfolio(portvals, risk_free_rate=0.0, sampling_freq=252.0):
    """
    Assess portfolio performance metrics.
    
    Parameters:
    -----------
    portvals : pd.DataFrame
        Portfolio values over time
    risk_free_rate : float, optional
        Daily risk-free rate, default 0
    sampling_freq : float, optional
        Trading days in a year, default 252
        
    Returns:
    --------
    tuple
        (Cumulative return, Average daily return, Standard deviation daily return,
         Sharpe ratio, End value)
    """
    # Get portfolio statistics (same as compute_portfolio_stats but with different return format)
    sr, cr, adr, sddr = compute_portfolio_stats(portvals, risk_free_rate, sampling_freq)
    
    # Calculate end value
    ev = portvals.iloc[-1, 0]
    
    return cr, adr, sddr, sr, ev


def compute_portfolio_stats(portfolio_values, daily_rf=0.0, samples_per_year=252):
    """
    Compute portfolio statistics: Sharpe ratio, cumulative returns, average daily return, and STD daily return.
    
    Parameters:
    -----------
    portfolio_values : pd.DataFrame
        Portfolio values over time
    daily_rf : float, optional
        Daily risk-free rate, default 0
    samples_per_year : int, optional
        Trading days in a year, default 252
        
    Returns:
    --------
    tuple
        (Sharpe ratio, Cumulative return, Average daily return, STD daily return)
    """
    # Calculate daily returns
    daily_returns = portfolio_values.pct_change().dropna()
    
    # Calculate average daily return and standard deviation of daily return
    avg_daily_ret = daily_returns.mean().values[0]
    std_daily_ret = daily_returns.std().values[0]
    
    # Calculate Sharpe ratio
    sharpe_ratio = np.sqrt(samples_per_year) * (avg_daily_ret - daily_rf) / std_daily_ret if std_daily_ret > 0 else 0
    
    # Calculate cumulative return
    cum_ret = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    cum_ret = cum_ret.values[0]
    
    return sharpe_ratio, cum_ret, avg_daily_ret, std_daily_ret