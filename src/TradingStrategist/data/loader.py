"""
Data loader utilities for TradingStrategist.

This module provides functions to load stock price data from CSV files,
plot financial data, and handle paths to various data files.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import os

def get_data_dir(config=None):
    """
    Return path to data directory.
    
    Parameters:
    -----------
    config : dict, optional
        Configuration dictionary that may contain data paths
        
    Returns:
    --------
    Path
        Path to the data directory
    """
    # If config has data_dir, use it
    if config is not None and 'data' in config and 'data_dir' in config['data']:
        data_dir = Path(config['data']['data_dir'])
    else:
        # Default to project's data directory
        data_dir = Path(__file__).parent.parent.parent.parent / "data"
        
    # Ensure directory exists
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
        
    return data_dir

def get_orders_data_file(config=None):
    """
    Return path to orders data file.
    
    Parameters:
    -----------
    config : dict, optional
        Configuration dictionary that may contain file paths
        
    Returns:
    --------
    Path
        Path to the orders data file
    """
    # If config has orders_file, use it
    if config is not None and 'data' in config and 'orders_file' in config['data']:
        return Path(config['data']['orders_file'])
    else:
        # Default to project's data directory
        base_dir = Path(__file__).parent.parent.parent.parent / "data"
        return base_dir / "orders.csv"

def get_learner_data_file(basefilename, config=None):
    """
    Return path for learner data file.
    
    Parameters:
    -----------
    basefilename : str
        Base file name to use (will add .csv)
    config : dict, optional
        Configuration dictionary that may contain data paths
        
    Returns:
    --------
    Path
        Path to the learner data file
    """
    # If config has learner_data_dir, use it
    if config is not None and 'data' in config and 'learner_data_dir' in config['data']:
        base_dir = Path(config['data']['learner_data_dir'])
    else:
        # Default to project's data/learner_data directory
        base_dir = Path(__file__).parent.parent.parent.parent / "data" / "learner_data"
        
    # Create directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)
        
    return base_dir / f"{basefilename}.csv"

def symbol_to_path(symbol, base_dir=None, config=None):
    """
    Return path to data file for a given stock symbol.
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    base_dir : str or Path, optional
        Base directory containing data files
    config : dict, optional
        Configuration dictionary that may contain data paths
        
    Returns:
    --------
    Path
        Path to the CSV file for the symbol
    """
    if base_dir is None:
        # If config has data_dir, use it
        if config is not None and 'data' in config and 'data_dir' in config['data']:
            base_dir = Path(config['data']['data_dir'])
        else:
            # Default to project's data directory
            base_dir = Path(__file__).parent.parent.parent.parent / "data"
    else:
        base_dir = Path(base_dir)
        
    # Ensure directory exists
    if not base_dir.exists():
        raise ValueError(f"Data directory not found: {base_dir}")
    
    return base_dir / f"{symbol}.csv"


def get_data(symbols, dates, addSPY=True, colname="Adj Close", config=None):
    """
    Read stock data for given symbols from CSV files.
    
    Parameters:
    -----------
    symbols : list
        List of symbols to read
    dates : pd.date_range
        Date range for which to get data
    addSPY : bool, optional
        Whether to add SPY data, default True
    colname : str, optional
        Column name to read from CSV, default "Adj Close"
    config : dict, optional
        Configuration dictionary that may contain data paths
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with stock data
    """
    # Add SPY if not already in the list and addSPY is True
    if addSPY and 'SPY' not in symbols:
        symbols = ['SPY'] + list(symbols)  # Ensure SPY is first
    
    # Create empty DataFrame with dates as index
    df = pd.DataFrame(index=dates)
    df.index.name = 'Date'
    
    for symbol in symbols:
        try:
            # Get the path for the symbol
            path = symbol_to_path(symbol, config=config)
            
            # Read data from CSV
            df_temp = pd.read_csv(
                path,
                index_col='Date',
                parse_dates=True,
                usecols=['Date', colname],
                na_values=['nan']
            )
            
            # Rename the column to match the symbol
            df_temp = df_temp.rename(columns={colname: symbol})
            
            # Join with main DataFrame
            df = df.join(df_temp)
        except FileNotFoundError:
            print(f"Warning: Data file for {symbol} not found, skipping.")
    
    # Handle NaN values
    df = df.dropna(subset=['SPY']) if addSPY else df.dropna()
    
    return df


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price", save_path=None, config=None):
    """
    Plot stock data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing stock data
    title : str, optional
        Plot title, default "Stock prices"
    xlabel : str, optional
        X-axis label, default "Date"
    ylabel : str, optional
        Y-axis label, default "Price"
    save_path : str, optional
        Path to save the plot, if None plot is displayed
    config : dict, optional
        Configuration dictionary that may contain plot settings
    """
    # Get figure size and style from config if available
    fig_size = (10, 6)
    if config and 'visualization' in config and 'fig_size' in config['visualization']:
        fig_size = tuple(config['visualization']['fig_size'])
    
    # Apply style from config
    if config and 'visualization' in config and 'style' in config['visualization']:
        plt.style.use(config['visualization']['style'])
    
    # Create plot
    plt.figure(figsize=fig_size)
    
    # Get colors from config or use default colormap
    if config and 'visualization' in config and 'colors' in config['visualization']:
        colors = config['visualization']['colors']
        # Plot each column with specified color if available
        for i, column in enumerate(df.columns):
            color = colors.get(column, None)  # Get color for symbol if defined
            if color:
                plt.plot(df.index, df[column], label=column, color=color)
            else:
                plt.plot(df.index, df[column], label=column)
    else:
        # Use default colormap
        for i, column in enumerate(df.columns):
            plt.plot(df.index, df[column], label=column)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()