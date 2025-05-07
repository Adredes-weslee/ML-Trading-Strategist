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
import json
from urllib.error import HTTPError

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

# Rate limiting configuration
INITIAL_DELAY = 2.0  # Starting delay between requests in seconds
MAX_DELAY = 60.0     # Maximum delay between requests
MAX_RETRIES = 3      # Maximum number of retries per ticker
BACKOFF_FACTOR = 2   # Factor by which to increase delay on failure

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

def test_yahoo_connection():
    """Test connection to Yahoo Finance API."""
    try:
        logging.info("Testing connection to Yahoo Finance...")
        response = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=1d")
        status_code = response.status_code
        logging.info(f"Yahoo Finance test connection status code: {status_code}")
        
        if status_code == 200:
            logging.info("Yahoo Finance API connection successful.")
            return True
        elif status_code == 429:
            logging.error("Yahoo Finance API returned 429 Too Many Requests - API rate limit exceeded.")
            logging.info("Waiting 60 seconds before continuing...")
            time.sleep(60)  # Wait a full minute
            return False
        else:
            logging.error(f"Yahoo Finance API returned unexpected status code: {status_code}")
            return False
    except Exception as e:
        logging.error(f"Error connecting to Yahoo Finance: {e}")
        return False

def download_with_retry(ticker, start_date, end_date):
    """Download stock data with exponential backoff retry logic."""
    yahoo_ticker = ticker
    # Convert special tickers
    if ticker == '$SPX':
        yahoo_ticker = '^GSPC'
    elif ticker == '$DJI':
        yahoo_ticker = '^DJI'
    elif ticker == '$VIX':
        yahoo_ticker = '^VIX'
    elif '.' in ticker:
        yahoo_ticker = ticker.replace('.', '-')
        
    delay = INITIAL_DELAY
    attempts = 0
    
    while attempts < MAX_RETRIES:
        try:
            stock_data = yf.download(yahoo_ticker, start=start_date, end=end_date, progress=False)
            return stock_data, None  # Success
        except HTTPError as e:
            if hasattr(e, 'code') and e.code == 429:
                attempts += 1
                wait_time = min(delay * (BACKOFF_FACTOR ** attempts), MAX_DELAY)
                logging.warning(f"{ticker}: Rate limited (429). Waiting {wait_time:.1f}s before retry {attempts}/{MAX_RETRIES}")
                time.sleep(wait_time)
                continue
            else:
                return None, f"HTTP error: {str(e)}"
        except Exception as e:
            if "429" in str(e):
                attempts += 1
                wait_time = min(delay * (BACKOFF_FACTOR ** attempts), MAX_DELAY)
                logging.warning(f"{ticker}: Rate limited (429). Waiting {wait_time:.1f}s before retry {attempts}/{MAX_RETRIES}")
                time.sleep(wait_time)
                continue
            else:
                return None, f"Error: {str(e)}"
    
    return None, f"Max retries ({MAX_RETRIES}) exceeded"

def download_stock_data(ticker, start_date='2000-01-01', end_date=None):
    """Download historical stock data for a specific ticker."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    file_path = DATA_DIR / f"{ticker}.csv"
    
    # Check if ticker is in our known delisted/changed list
    if ticker in DELISTED_TICKERS:
        logging.info(f"{ticker}: Skipping - {DELISTED_TICKERS[ticker]}")
        return "skipped"
    
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
        
        # Download the data with retry logic
        stock_data, error = download_with_retry(ticker, start_date, end_date)
        
        # If error occurred or no data was returned
        if error or stock_data is None or stock_data.empty:
            if error:
                logging.error(f"{ticker}: Download failed - {error}")
                if "429" in str(error):
                    return "rate-limited"
                return "failed"
            else:
                logging.warning(f"{ticker}: No data available for the specified date range.")
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
        time.sleep(3.0)  # Extra delay for index data
    
    return status

def main():
    """Main function to download all S&P 500 stock data."""
    logging.info("Starting S&P 500 data download process...")
    logging.info(f"Data will be saved to: {DATA_DIR}")
    
    # First test connection to Yahoo Finance
    if not test_yahoo_connection():
        logging.error("Failed to connect to Yahoo Finance API. Please check your internet connection and try again later.")
        print("ERROR: Could not connect to Yahoo Finance. See log for details.")
        return
    
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
        "rate-limited": 0,
        "error": 0
    }
    
    # Rate limiting tracking
    consecutive_rate_limits = 0
    rate_limit_pause_threshold = 3  # How many consecutive rate limits before pausing
    
    # Download stock data for each ticker
    for i, ticker in enumerate(all_tickers):
        try:
            # Show progress in console
            if i % progress_interval == 0:
                sys.stdout.write("=")
                sys.stdout.flush()
                
            # Check if we need to pause due to rate limiting
            if consecutive_rate_limits >= rate_limit_pause_threshold:
                pause_time = 120  # 2 minutes
                logging.warning(f"Detected {consecutive_rate_limits} consecutive rate limits. Pausing for {pause_time} seconds...")
                print(f"\nRate limit detected - pausing for {pause_time} seconds...", end="")
                sys.stdout.flush()
                time.sleep(pause_time)
                consecutive_rate_limits = 0
                print(" resuming")
            
            result = download_stock_data(ticker)
            status_counts[result] += 1
            
            # Track consecutive rate limits
            if result == "rate-limited":
                consecutive_rate_limits += 1
            else:
                consecutive_rate_limits = 0
                
        except Exception as e:
            logging.error(f"Unexpected error processing {ticker}: {str(e)}")
            status_counts["error"] += 1
        
        # Add a variable delay to avoid hitting API limits
        # Use longer delays when we're seeing rate limits
        if consecutive_rate_limits > 0:
            delay = 4.0 + (random.random() * 2.0)  # 4-6 seconds
        else:
            delay = 2.0 + (random.random() * 1.0)  # 2-3 seconds
            
        time.sleep(delay)
    
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
    logging.info(f"Rate limited: {status_counts['rate-limited']}")
    logging.info(f"Failed downloads: {status_counts['failed']}")
    logging.info(f"Errors: {status_counts['error']}")
    
    # Print index data status
    logging.info("\n=== INDEX DATA STATUS ===")
    for index, status in index_results.items():
        logging.info(f"{index}: {status}")
    
    logging.info("S&P 500 data download completed.")
    
    if status_counts['rate-limited'] > 0:
        logging.warning("\nWARNING: Your downloads were rate-limited by Yahoo Finance.")
        logging.warning("Try running the script again later or with a smaller number of tickers.")
        print("\nWARNING: Rate limits encountered - see log for details.")

if __name__ == "__main__":
    main()