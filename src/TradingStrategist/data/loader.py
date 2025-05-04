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
    # Get data directory from config, environment variable, or default
    if config is not None and 'data' in config and 'base_dir' in config['data']:
        base_dir = config['data']['base_dir']
    elif base_dir is None:
        # Try environment variable first
        base_dir = os.environ.get("MARKET_DATA_DIR")
        
        if base_dir is None:
            # Try to find the data directory relative to the project root
            current_file_path = Path(__file__).resolve()
            project_root = current_file_path.parent.parent.parent.parent
            data_dir = project_root / "data"
            
            if data_dir.exists():
                base_dir = str(data_dir)
            else:
                # Fallback to relative path from current working directory
                base_dir = "./data"
    
    # Convert to Path object for easier path manipulation
    base_path = Path(base_dir)
    file_path = base_path / f"{symbol}.csv"
    
    return str(file_path)


def get_data(symbols, dates, addSPY=True, colname="Adj Close", config=None):
    """
    Read stock data for given symbols from CSV files and return a pandas DataFrame.
    
    Parameters:
    -----------
    symbols : list
        List of stock symbols to read
    dates : pd.date_range
        Range of dates to retrieve data for
    addSPY : bool, optional
        Whether to add SPY data if not already in symbols, default True
    colname : str, optional
        Column to read from CSV files, default "Adj Close"
    config : dict, optional
        Configuration dictionary
        
    Returns:
    --------
    pd.DataFrame: Stock data for the given symbols and dates
    """
    # Create empty DataFrame with dates as index
    df_data = pd.DataFrame(index=dates)
    
    # Add SPY if needed
    symbols_list = symbols.copy()
    if addSPY and 'SPY' not in symbols_list:
        symbols_list = ['SPY'] + symbols_list
    
    # Read data for each symbol
    for symbol in symbols_list:
        file_path = symbol_to_path(symbol, config=config)
        try:
            df_temp = pd.read_csv(file_path, index_col='Date', parse_dates=True, usecols=['Date', colname])
            # Rename column to symbol for identification
            df_temp = df_temp.rename(columns={colname: symbol})
            df_data = df_data.join(df_temp, how='left')
        except FileNotFoundError:
            raise FileNotFoundError(f"Price data not found for {symbol} at {file_path}. Please download the data first.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"No data in file for {symbol} at {file_path}.")
        except Exception as e:
            raise Exception(f"Error loading data for {symbol}: {str(e)}")
    
    # Handle missing data
    if df_data.isna().any().any():
        print("Warning: Missing data detected. Forward-filling missing values.")
        # Update to use ffill() and bfill() instead of fillna(method='ffill')
        df_data = df_data.ffill()
        df_data = df_data.bfill()
        if df_data.isna().any().any():
            raise ValueError("Missing data could not be filled completely. Check your input date range.")
    
    return df_data


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price", save_path=None, config=None):
    """Plot stock prices."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    
    for column in df.columns:
        ax.plot(df.index, df[column], label=column)
    
    ax.legend(loc='best')
    
    if save_path is not None:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    return fig, ax


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