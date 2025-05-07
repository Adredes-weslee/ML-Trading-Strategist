#!/usr/bin/env python
"""
Download S&P 500 stock data from Yahoo Finance.
This script fetches historical data for all S&P 500 companies from 2000 to the present.
"""

import os
import time
from datetime import datetime
import pandas as pd
import yfinance as yf
import requests
import io
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download_data.log"),
        logging.StreamHandler()
    ]
)

# Create data directory if it doesn't exist
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def get_sp500_tickers():
    """Get current S&P 500 tickers using Wikipedia."""
    logging.info("Fetching current S&P 500 tickers from Wikipedia...")
    
    try:
        table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = table[0]
        return df['Symbol'].str.replace('.', '-').tolist()
    except Exception as e:
        logging.error(f"Error fetching S&P 500 tickers: {e}")
        return []

def get_historical_sp500_tickers():
    """Get historical S&P 500 constituents that may no longer be in the index."""
    # This is a simplified approach - ideally we would source a complete historical list
    # For now, we'll scan the data directory to find existing tickers
    logging.info("Checking for existing stock data files...")
    
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
    
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    
    try:
        # If file exists, only download new data
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path)
            if not existing_data.empty:
                # Get the last date in the existing data
                last_date = pd.to_datetime(existing_data['Date']).max().strftime('%Y-%m-%d')
                # Only download data after the last date
                if last_date >= end_date:
                    logging.info(f"{ticker}: Already up-to-date, skipping.")
                    return
                start_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)
                start_date = start_date.strftime('%Y-%m-%d')
                logging.info(f"{ticker}: Updating data from {start_date} to {end_date}")
        else:
            logging.info(f"{ticker}: Downloading full history from {start_date} to {end_date}")
        
        # Download the data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        # If no data was returned, log and return
        if stock_data.empty:
            logging.warning(f"{ticker}: No data available for the specified date range.")
            return
        
        # Reset index to make Date a column
        stock_data.reset_index(inplace=True)
        
        # If we're updating existing data, append the new data
        if os.path.exists(file_path) and 'existing_data' in locals():
            # Convert dates to datetime for proper comparison
            existing_data['Date'] = pd.to_datetime(existing_data['Date'])
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            
            # Remove any overlap
            stock_data = stock_data[~stock_data['Date'].isin(existing_data['Date'])]
            
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
    
    except Exception as e:
        logging.error(f"{ticker}: Error downloading data - {str(e)}")

def download_index_data():
    """Download major index data for comparison."""
    indices = {
        '$SPX': '^GSPC',  # S&P 500
        '$DJI': '^DJI',   # Dow Jones Industrial Average
        '$VIX': '^VIX'    # Volatility Index
    }
    
    for name, symbol in indices.items():
        logging.info(f"Downloading {name} ({symbol}) data...")
        download_stock_data(name, start_date='2000-01-01')

def main():
    """Main function to download all S&P 500 stock data."""
    logging.info("Starting S&P 500 data download process...")
    
    # Get current S&P 500 tickers
    current_tickers = get_sp500_tickers()
    logging.info(f"Found {len(current_tickers)} current S&P 500 tickers.")
    
    # Get historical tickers
    historical_tickers = get_historical_sp500_tickers()
    logging.info(f"Found {len(historical_tickers)} historical stock data files.")
    
    # Combine and deduplicate tickers
    all_tickers = list(set(current_tickers + historical_tickers))
    logging.info(f"Preparing to download/update data for {len(all_tickers)} stocks.")
    
    # Download index data first
    download_index_data()
    
    # Download stock data for each ticker
    for i, ticker in enumerate(all_tickers):
        logging.info(f"Processing {ticker} ({i+1}/{len(all_tickers)})")
        download_stock_data(ticker)
        # Add a small delay to avoid hitting API limits
        time.sleep(0.5)
    
    logging.info("S&P 500 data download completed.")

if __name__ == "__main__":
    main()