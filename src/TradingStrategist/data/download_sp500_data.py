#!/usr/bin/env python
"""
Download S&P 500 stock data from Yahoo Finance.
This script fetches historical data for all S&P 500 companies from 2000 to the present.
"""

import os
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import yfinance as yf
import requests
import io
import logging
import random
import sys

# Get project root directory (3 levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
# Set data directory to be in project root
DATA_DIR = PROJECT_ROOT / "data"
# Set log file in project root
LOG_FILE = PROJECT_ROOT / "download_data.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Known tickers that have been delisted or changed
DELISTED_TICKERS = {
    "NOVL": "Novell (acquired by Attachmate)",
    "GENZ": "Genzyme (acquired by Sanofi)",
    "DDR": "DDR Corp (renamed SITE Centers - SITC)",
    "JNY": "Jones Group (acquired by Sycamore Partners)",
    "TEG": "Integrys Energy Group (merged with WEC Energy)",
    "PLTR": "Palantir (IPO was after 2012)",
    "WAMUQ": "Washington Mutual (bankrupt in 2008)",
    "KBH": "KB Home (symbol changed)",
    "NEE": "NextEra Energy (symbol changed from FPL)",
    "PTV": "Pactiv (acquired by Reynolds Group)",
    "AMT": "American Tower (converted to REIT)",
    # Add more as they're identified
}

def get_sp500_tickers():
    """Get current S&P 500 tickers using Wikipedia."""
    logging.info(f"Fetching current S&P 500 tickers from Wikipedia...")
    
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        return df['Symbol'].str.replace('.', '-').tolist()
    except Exception as e:
        logging.error(f"Error fetching S&P 500 tickers: {e}")
        return []

def get_historical_sp500_tickers():
    """Get historical S&P 500 constituents that may no longer be in the index."""
    logging.info(f"Checking for existing stock data files in {DATA_DIR}...")
    
    existing_tickers = []
    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv'):
            ticker = file.replace('.csv', '')
            if ticker not in ['$DJI', '$SPX', '$VIX']:  # Skip indices
                existing_tickers.append(ticker)
    
    return existing_tickers

def download_stock_data(ticker, start_date='2000-01-01', end_date=None):
    """Download historical stock data for a specific ticker."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    file_path = DATA_DIR / f"{ticker}.csv"
    
    # Check if ticker is in our known delisted/changed list
    if ticker in DELISTED_TICKERS:
        logging.info(f"{ticker}: Skipping - {DELISTED_TICKERS[ticker]}")
        return "skipped"
    
    # Yahoo Finance uses different symbols for some tickers
    # Map special cases
    yahoo_ticker = ticker
    if ticker == '$SPX':
        yahoo_ticker = '^GSPC'
    elif ticker == '$DJI':
        yahoo_ticker = '^DJI'
    elif ticker == '$VIX':
        yahoo_ticker = '^VIX'
    elif '.' in ticker:
        yahoo_ticker = ticker.replace('.', '-')
    
    try:
        # If file exists, only download new data
        if file_path.exists():
            existing_data = pd.read_csv(file_path)
            if not existing_data.empty:
                # Get the last date in the existing data
                last_date = pd.to_datetime(existing_data['Date']).max().strftime('%Y-%m-%d')
                # Only download data after the last date
                if last_date >= end_date:
                    logging.info(f"{ticker}: Already up-to-date, skipping.")
                    return "up-to-date"
                start_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)
                start_date = start_date.strftime('%Y-%m-%d')
                logging.info(f"{ticker}: Updating data from {start_date} to {end_date}")
        else:
            logging.info(f"{ticker}: Downloading full history from {start_date} to {end_date}")
        
        # Download the data with progress suppressed
        try:
            stock_data = yf.download(yahoo_ticker, start=start_date, end=end_date, progress=False)
        except Exception as e:
            logging.error(f"Failed to download {ticker}: {str(e)}")
            return "failed"
        
        # If no data was returned, log and return
        if stock_data.empty:
            logging.warning(f"{ticker}: No data available for the specified date range.")
            # Don't consider this a failure as the ticker may be delisted
            return "no-data"
        
        # Reset index to make Date a column
        stock_data.reset_index(inplace=True)
        
        # If we're updating existing data, append the new data
        if file_path.exists() and 'existing_data' in locals():
            # Convert dates to datetime for proper comparison
            existing_data['Date'] = pd.to_datetime(existing_data['Date'])
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            
            # Remove any overlap
            stock_data = stock_data[~stock_data['Date'].isin(existing_data['Date'])]
            
            if stock_data.empty:
                logging.info(f"{ticker}: No new data to add.")
                return "no-new-data"
                
            # Combine the dataframes
            combined_data = pd.concat([existing_data, stock_data])
            
            # Sort by date
            combined_data = combined_data.sort_values('Date')
            
            # Convert Date back to string format
            combined_data['Date'] = combined_data['Date'].dt.strftime('%Y-%m-%d')
            
            # Save the combined data
            combined_data.to_csv(file_path, index=False)
            logging.info(f"{ticker}: Updated with {len(stock_data)} new data points.")
        else:
            # Save as new file
            stock_data.to_csv(file_path, index=False)
            logging.info(f"{ticker}: Downloaded {len(stock_data)} data points.")
        
        return "success"
    
    except Exception as e:
        logging.error(f"{ticker}: Error downloading data - {str(e)}")
        return "error"

def download_index_data():
    """Download major index data for comparison."""
    indices = {
        '$SPX': '^GSPC',  # S&P 500
        '$DJI': '^DJI',   # Dow Jones Industrial Average
        '$VIX': '^VIX'    # Volatility Index
    }
    
    status = {}
    for name, symbol in indices.items():
        logging.info(f"Downloading {name} ({symbol}) data...")
        status[name] = download_stock_data(name, start_date='2000-01-01')
        time.sleep(1.5)  # Increased delay for index data
    
    return status

def main():
    """Main function to download all S&P 500 stock data."""
    logging.info("Starting S&P 500 data download process...")
    logging.info(f"Data will be saved to: {DATA_DIR}")
    
    # Get current S&P 500 tickers
    current_tickers = get_sp500_tickers()
    logging.info(f"Found {len(current_tickers)} current S&P 500 tickers.")
    
    # Get historical tickers
    historical_tickers = get_historical_sp500_tickers()
    logging.info(f"Found {len(historical_tickers)} historical stock data files.")
    
    # Combine and deduplicate tickers
    all_tickers = list(set(current_tickers + historical_tickers))
    all_tickers.sort()  # Sort for more readable output
    total_tickers = len(all_tickers)
    logging.info(f"Preparing to download/update data for {total_tickers} stocks.")
    
    # Add progress bar for console
    print(f"Processing {total_tickers} tickers [", end="")
    progress_interval = max(1, total_tickers // 50)  # Show 50 progress marks
    
    # Download index data first
    logging.info("Downloading index data first...")
    index_results = download_index_data()
    
    # Track status counts
    status_counts = {
        "success": 0,
        "up-to-date": 0,
        "skipped": 0,
        "no-data": 0,
        "no-new-data": 0,
        "failed": 0,
        "error": 0
    }
    
    # Download stock data for each ticker
    for i, ticker in enumerate(all_tickers):
        try:
            result = download_stock_data(ticker)
            status_counts[result] += 1
            
            # Show progress in console
            if i % progress_interval == 0:
                sys.stdout.write("=")
                sys.stdout.flush()
        except Exception as e:
            logging.error(f"Unexpected error processing {ticker}: {str(e)}")
            status_counts["error"] += 1
        
        # Add a variable delay to avoid hitting API limits (1-2 seconds)
        time.sleep(1.0 + random.random())
    
    # Complete the progress bar
    print("] Done!")
    
    # Print summary
    logging.info("\n=== DOWNLOAD SUMMARY ===")
    logging.info(f"Total tickers processed: {total_tickers}")
    logging.info(f"Successfully downloaded new data: {status_counts['success']}")
    logging.info(f"Already up-to-date: {status_counts['up-to-date']}")
    logging.info(f"No new data to add: {status_counts['no-new-data']}")
    logging.info(f"Skipped known delisted/changed tickers: {status_counts['skipped']}")
    logging.info(f"No data available (likely delisted): {status_counts['no-data']}")
    logging.info(f"Failed downloads: {status_counts['failed']}")
    logging.info(f"Errors: {status_counts['error']}")
    
    # Print index data status
    logging.info("\n=== INDEX DATA STATUS ===")
    for index, status in index_results.items():
        logging.info(f"{index}: {status}")
    
    logging.info("S&P 500 data download completed.")

if __name__ == "__main__":
    main()