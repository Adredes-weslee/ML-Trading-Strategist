"""
Data loader utilities for TradingStrategist.

This module provides functions to load stock price data from CSV files,
plot financial data, and handle paths to various data files.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def symbol_to_path(symbol, base_dir=None, config=None):
    """
    Return CSV file path given ticker symbol.
    
    Parameters:
    -----------
    symbol : str
        The ticker symbol for the stock
    base_dir : str, optional
        Base directory for data files
    config : dict, optional
        Configuration dictionary
        
    Returns:
    --------
    str
        Path to the CSV file for the symbol
    """
    if config is not None and 'data' in config and 'base_dir' in config['data']:
        base_dir = config['data']['base_dir']
    elif base_dir is None:
        base_dir = os.environ.get("MARKET_DATA_DIR", "../data/")
    return os.path.join(base_dir, f"{symbol}.csv")


def get_data(symbols, dates, addSPY=True, colname="Adj Close", config=None):
    """
    Read stock data (adjusted close) for given symbols from CSV files.
    
    Parameters:
    -----------
    symbols : list
        List of symbols to retrieve data for
    dates : pd.DatetimeIndex
        Dates to retrieve data for
    addSPY : bool, optional
        Add SPY data for reference
    colname : str, optional
        Column name to use from the CSV files
    config : dict, optional
        Configuration dictionary
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing data for all symbols
    """
    # Use config if provided
    if config is not None and 'data' in config:
        if 'add_spy' in config['data']:
            addSPY = config['data']['add_spy']
        if 'column_name' in config['data']:
            colname = config['data']['column_name']
    
    df = pd.DataFrame(index=dates)
    
    # Add SPY for reference, if absent
    if addSPY and "SPY" not in symbols:  
        symbols = ["SPY"] + list(symbols)
    
    for symbol in symbols:
        df_temp = pd.read_csv(
            symbol_to_path(symbol, config=config),
            index_col="Date",
            parse_dates=True,
            usecols=["Date", colname],
            na_values=["nan"],
        )
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        
        # Drop dates SPY did not trade
        if symbol == "SPY":
            df = df.dropna(subset=["SPY"])
    
    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price", save_path=None, config=None):
    """
    Plot stock prices with a custom title and meaningful axis labels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing stock price data
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    save_path : str, optional
        Path to save the figure instead of displaying it
    config : dict, optional
        Configuration dictionary
    """
    # Use config if provided
    fig_size = (10, 6)
    font_size = 12
    
    if config is not None and 'visualization' in config:
        if 'title' in config['visualization'] and title == "Stock prices":
            title = config['visualization']['title']
        if 'xlabel' in config['visualization']:
            xlabel = config['visualization']['xlabel']
        if 'ylabel' in config['visualization']:
            ylabel = config['visualization']['ylabel']
        if 'save_path' in config['visualization'] and not save_path:
            save_path = config['visualization']['save_path']
        if 'fig_size' in config['visualization']:
            fig_size = tuple(config['visualization']['fig_size'])
        if 'font_size' in config['visualization']:
            font_size = config['visualization']['font_size']
    
    fig, ax = plt.subplots(figsize=fig_size)
    df.plot(ax=ax, title=title, fontsize=font_size)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def get_data_dir(folder_name, default_folder, config=None):
    """
    Get the directory for a specific type of data.
    
    Parameters:
    -----------
    folder_name : str
        Environment variable name for the folder
    default_folder : str
        Default folder name if environment variable not set
    config : dict, optional
        Configuration dictionary
        
    Returns:
    --------
    Path
        Path object to the data directory
    """
    # Use config if provided
    config_folder = None
    if config is not None and 'data' in config and 'directories' in config['data']:
        if folder_name in config['data']['directories']:
            config_folder = config['data']['directories'][folder_name]
    
    data_dir = os.environ.get(folder_name, config_folder or default_folder)
    path = Path(data_dir)
    os.makedirs(path, exist_ok=True)
    return path


def get_orders_data_file(basefilename, config=None):
    """
    Get a file handle for an orders data file.
    
    Parameters:
    -----------
    basefilename : str
        Base filename of the orders file
    config : dict, optional
        Configuration dictionary
        
    Returns:
    --------
    file
        Open file handle
    """
    orders_dir = get_data_dir("ORDERS_DATA_DIR", "orders/", config)
    return open(os.path.join(orders_dir, basefilename))


def get_learner_data_file(basefilename, config=None):
    """
    Get a file handle for a learner data file.
    
    Parameters:
    -----------
    basefilename : str
        Base filename of the learner data file
    config : dict, optional
        Configuration dictionary
        
    Returns:
    --------
    file
        Open file handle for reading
    """
    learner_dir = get_data_dir("LEARNER_DATA_DIR", "data/", config)
    return open(os.path.join(learner_dir, basefilename), "r")