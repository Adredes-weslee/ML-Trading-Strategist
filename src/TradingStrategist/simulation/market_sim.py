"""
Market Simulator

This module provides functionality for simulating market trading
and evaluating trading strategy performance.
"""

import pandas as pd
import numpy as np


def compute_portvals(orders, start_val=100000.0, commission=9.95, impact=0.005,
                     start_date=None, end_date=None, prices_df=None, 
                     cost_model='fixed_plus_slippage', slippage_std=0.0):
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
    cost_model : str, optional
        Transaction cost model to use: 'fixed_only', 'fixed_plus_slippage',
        'volume_adjusted', or 'proportional', default 'fixed_plus_slippage'
    slippage_std : float, optional
        Standard deviation for random slippage component, default 0.0
        
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
    
    # Validate cost model
    valid_models = ['fixed_only', 'fixed_plus_slippage', 'volume_adjusted', 'proportional']
    if cost_model not in valid_models:
        raise ValueError(f"Invalid cost_model: {cost_model}. Must be one of {valid_models}")
    
    # Process orders and update holdings and cash
    for date in orders.index:
        if date not in prices_all.index:
            continue
        
        for symbol in symbols:
            shares = orders.loc[date, symbol]
            if shares == 0:
                continue
            
            # Calculate transaction cost based on selected model
            price = prices_all.loc[date, symbol]
            trade_value = abs(shares) * price
            
            # Apply random slippage if configured (simulates market noise)
            if slippage_std > 0:
                # Random normal noise proportional to price and configured std
                random_slippage = np.random.normal(0, slippage_std * price)
                # Adds noise in the adverse direction of the trade
                price_with_slippage = price + (np.sign(shares) * random_slippage)
                price = max(0.01, price_with_slippage)  # Ensure price doesn't go negative
            
            if cost_model == 'fixed_only':
                # Fixed commission only
                transaction_cost = commission if shares != 0 else 0
                effective_price = price  # No additional impact
            
            elif cost_model == 'fixed_plus_slippage':
                # Fixed commission plus market impact (default)
                market_impact_cost = abs(shares) * price * impact
                transaction_cost = commission + market_impact_cost
                effective_price = price * (1 + impact * np.sign(shares))
            
            elif cost_model == 'volume_adjusted':
                # Impact increases with trade size (sqrt model - common in literature)
                # sqrt(shares) reflects that doubling order size doesn't double market impact
                market_impact_factor = impact * np.sqrt(abs(shares) / 100)  # 100 shares as baseline
                market_impact_cost = abs(shares) * price * market_impact_factor
                transaction_cost = commission + market_impact_cost
                effective_price = price * (1 + market_impact_factor * np.sign(shares))
            
            elif cost_model == 'proportional':
                # Commission is percentage of trade value
                transaction_cost = trade_value * impact
                effective_price = price * (1 + impact * np.sign(shares))
            
            else:
                # Should never reach here due to validation above
                transaction_cost = commission
                effective_price = price
            
            # Update holdings
            holdings.loc[date:, symbol] += float(shares)
            
            # Update cash - use effective price for calculating cash impact
            actual_trade_value = shares * effective_price
            cash_value = cash[date] - (actual_trade_value + commission)
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