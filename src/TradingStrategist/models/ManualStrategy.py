"""
Manual Strategy - Rule-based trading strategy for comparison with machine learning approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from TradingStrategist.indicators.technical import Indicators
from TradingStrategist.data.loader import get_data
from TradingStrategist.utils.helpers import get_output_path
import os

class ManualStrategy:
    """Manual rule-based trading strategy using technical indicators."""
        
    def __init__(self, verbose=False, thresholds=None, position_size=1000, indicators_config=None,
             window_size=None, rsi_window=None, stoch_window=None, cci_window=None,
             buy_threshold=None, sell_threshold=None, indicator_thresholds=None, **kwargs):
        """
        Initialize the ManualStrategy.
        
        Parameters:
        -----------
        verbose : bool, optional
            Whether to output additional information, default False
        thresholds : dict, optional
            Custom thresholds for technical indicators
        position_size : int, optional
            Number of shares to trade, default 1000
        indicators_config : dict, optional
            Configuration dictionary for technical indicators
        window_size, rsi_window, etc.: Individual parameter overrides
        indicator_thresholds : dict, optional
            Dictionary of detailed indicator thresholds
        **kwargs : dict
            Additional parameters not explicitly listed
        """
        self.verbose = verbose
        self.position_size = position_size
        self.indicators = Indicators(config=indicators_config)
        
        # Default thresholds (standard values from technical analysis)
        self.thresholds = {
            'bollinger_upper': 1.0,
            'bollinger_lower': -1.0,
            'rsi_upper': 70,
            'rsi_lower': 30,
            'stoch_upper': 80,
            'stoch_lower': 20,
            'cci_upper': 100,
            'cci_lower': -100,
            'min_vote_buy': 3,
            'min_vote_sell': 3,
            'window_size': 20,
            'rsi_window': 14,
            'stoch_window': 14,
            'cci_window': 20
        }
        
        # Override defaults with custom thresholds if provided
        if thresholds is not None:
            self.thresholds.update(thresholds)
        
        # Handle indicator_thresholds (detailed thresholds)
        if indicator_thresholds is not None:
            self.thresholds.update(indicator_thresholds)
        
        # Handle individual parameter overrides - direct parameters take precedence
        param_mapping = {
            'window_size': window_size,
            'rsi_window': rsi_window, 
            'stoch_window': stoch_window,
            'cci_window': cci_window,
        }
        
        # Convert buy_threshold and sell_threshold to voting thresholds if provided
        # This handles the simplified threshold system in the yaml files
        if buy_threshold is not None and sell_threshold is not None:
            # Assuming 5 indicators total, convert percentage threshold to vote count
            # For example, buy_threshold=0.02 means at least 60% of indicators must agree
            min_votes_buy = max(1, round(5 * (0.5 + buy_threshold)))
            min_votes_sell = max(1, round(5 * (0.5 + abs(sell_threshold))))
            param_mapping['min_vote_buy'] = min_votes_buy
            param_mapping['min_vote_sell'] = min_votes_sell
        
        # Update thresholds with any non-None parameters
        for key, value in param_mapping.items():
            if value is not None:
                self.thresholds[key] = value
                
        # Add any other kwargs to thresholds (for extensibility)
        for key, value in kwargs.items():
            if key not in ['verbose', 'position_size', 'indicators_config']:
                self.thresholds[key] = value
    
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), 
                  ed=dt.datetime(2009, 12, 31), sv=100000):
        """
        Test trading policy and generate trades.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol to trade
        sd : datetime
            Start date
        ed : datetime
            End date
        sv : int
            Starting portfolio value
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing trades (buy/sell/hold)
        """
        # Get price data - need to add a buffer before start date for indicators
        buffer_days = max(
            self.thresholds['window_size'],
            self.thresholds['rsi_window'],
            self.thresholds['stoch_window'],
            self.thresholds['cci_window']
        ) + 10  # Extra buffer for safety
        
        buffer_sd = sd - dt.timedelta(days=buffer_days)
        prices_with_buffer = get_data([symbol], pd.date_range(buffer_sd, ed))
        if 'SPY' in prices_with_buffer.columns:
            prices_with_buffer = prices_with_buffer.drop(columns=['SPY'])
        
        # Compute technical indicators
        indicators = self._compute_indicators(prices_with_buffer[symbol])
        
        # Generate trading signals based on indicators
        signals = self._generate_signals(indicators)
        
        # Convert signals to trades
        trades = self._generate_trades(signals, prices_with_buffer.index, symbol)
        
        # Filter trades to match the requested date range
        trades = trades[trades.index >= sd]
        
        if self.verbose:
            print(f"Generated {(trades != 0).sum()[0]} trade signals")
            
        return trades
    
    def _compute_indicators(self, prices):
        """
        Compute technical indicators for the given price series.
        
        Parameters:
        -----------
        prices : pandas.Series
            Price data for a single symbol
            
        Returns:
        --------
        dict
            Dictionary of computed indicators
        """
        window_size = self.thresholds['window_size']
        rsi_window = self.thresholds['rsi_window']
        stoch_window = self.thresholds['stoch_window']
        cci_window = self.thresholds['cci_window']
        
        indicators = {}
        
        # Compute Bollinger Bands
        indicators['bollinger'] = self.indicators.bollinger_percent_indicator(prices, window=window_size)
        
        # Compute RSI
        indicators['rsi'] = self.indicators.rsi_indicator(prices, window=rsi_window)
        
        # Compute MACD
        macd, signal = self.indicators.macd_indicator(prices)
        indicators['macd'] = macd - signal  # MACD histogram
        
        # Compute Stochastic Oscillator - FIX: match method name with Indicators class
        indicators['stoch'] = self.indicators.stochastic_indicator(prices, window=stoch_window)
        
        # Compute CCI
        indicators['cci'] = self.indicators.cci_indicator(prices, window=cci_window)
        
        return indicators
    
    def _generate_signals(self, indicators):
        """
        Generate buy/sell/hold signals based on technical indicators.
        
        Parameters:
        -----------
        indicators : dict
            Dictionary of computed indicators
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with buy/sell/hold signals
        """
        # Extract thresholds
        bollinger_upper = self.thresholds['bollinger_upper']
        bollinger_lower = self.thresholds['bollinger_lower']
        rsi_upper = self.thresholds['rsi_upper']
        rsi_lower = self.thresholds['rsi_lower']
        stoch_upper = self.thresholds['stoch_upper']
        stoch_lower = self.thresholds['stoch_lower']
        cci_upper = self.thresholds['cci_upper']
        cci_lower = self.thresholds['cci_lower']
        min_vote_buy = self.thresholds['min_vote_buy']
        min_vote_sell = self.thresholds['min_vote_sell']
        
        # Generate individual indicator signals
        bollinger_buy = indicators['bollinger'] <= bollinger_lower
        bollinger_sell = indicators['bollinger'] >= bollinger_upper
        
        rsi_buy = indicators['rsi'] <= rsi_lower
        rsi_sell = indicators['rsi'] >= rsi_upper
        
        macd_buy = indicators['macd'] > 0
        macd_sell = indicators['macd'] < 0
        
        stoch_buy = indicators['stoch'] <= stoch_lower
        stoch_sell = indicators['stoch'] >= stoch_upper
        
        cci_buy = indicators['cci'] <= cci_lower
        cci_sell = indicators['cci'] >= cci_upper
        
        # Count buy and sell votes
        buy_votes = bollinger_buy.astype(int) + rsi_buy.astype(int) + \
                   macd_buy.astype(int) + stoch_buy.astype(int) + cci_buy.astype(int)
                   
        sell_votes = bollinger_sell.astype(int) + rsi_sell.astype(int) + \
                    macd_sell.astype(int) + stoch_sell.astype(int) + cci_sell.astype(int)
        
        # Generate final signals
        signals = pd.DataFrame(index=indicators['bollinger'].index)
        signals['signal'] = 0  # Default to hold
        
        signals.loc[buy_votes >= min_vote_buy, 'signal'] = 1  # Buy
        signals.loc[sell_votes >= min_vote_sell, 'signal'] = -1  # Sell
        
        return signals
    
    def _generate_trades(self, signals, dates, symbol):
        """
        Convert signals to actual trades, considering current position.
        
        Parameters:
        -----------
        signals : pandas.DataFrame
            DataFrame with buy/sell/hold signals
        dates : pandas.DatetimeIndex
            Dates for the trading period
        symbol : str
            Stock symbol
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with trades (positive for buy, negative for sell)
        """
        trades = pd.DataFrame(0, index=dates, columns=[symbol])
        
        position = 0  # Start with no position
        
        for date in signals.index:
            signal = signals.loc[date, 'signal']
            
            if signal == 1 and position <= 0:  # Buy signal
                if position == 0:
                    trades.loc[date, symbol] = self.position_size
                    position = 1
                elif position == -1:
                    trades.loc[date, symbol] = 2 * self.position_size  # Exit short and go long
                    position = 1
            
            elif signal == -1 and position >= 0:  # Sell signal
                if position == 0:
                    trades.loc[date, symbol] = -self.position_size
                    position = -1
                elif position == 1:
                    trades.loc[date, symbol] = -2 * self.position_size  # Exit long and go short
                    position = -1
        
        return trades
    
    def plot_indicators_with_trades(self, symbol, sd, ed, trades=None, output_file=None):
        """
        Plot all indicators with trade markers.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        sd : datetime
            Start date
        ed : datetime
            End date
        trades : pandas.DataFrame, optional
            DataFrame with trades. If None, will compute trades.
        output_file : str, optional
            Output file path for saving plot
        """
        # Get price data
        prices = get_data([symbol], pd.date_range(sd, ed))
        if 'SPY' in prices.columns:
            prices = prices.drop(columns=['SPY'])
        
        # Get or compute trades
        if trades is None:
            trades = self.testPolicy(symbol, sd, ed)
        
        # Compute indicators
        indicators = self._compute_indicators(prices[symbol])
        
        # Set up the plot
        fig, axes = plt.subplots(6, 1, figsize=(12, 15), sharex=True)
        
        # Plot price
        axes[0].plot(prices.index, prices[symbol], label='Price', color='black')
        axes[0].set_title(f'{symbol} Price and Indicators')
        axes[0].legend()
        
        # Plot Bollinger Bands %
        axes[1].plot(indicators['bollinger'].index, indicators['bollinger'], label='BB%', color='blue')
        axes[1].axhline(y=self.thresholds['bollinger_upper'], color='red', linestyle='--', label='Upper Threshold')
        axes[1].axhline(y=self.thresholds['bollinger_lower'], color='green', linestyle='--', label='Lower Threshold')
        axes[1].set_title('Bollinger Bands %')
        axes[1].legend()
        
        # Plot RSI
        axes[2].plot(indicators['rsi'].index, indicators['rsi'], label='RSI', color='purple')
        axes[2].axhline(y=self.thresholds['rsi_upper'], color='red', linestyle='--', label='Upper Threshold')
        axes[2].axhline(y=self.thresholds['rsi_lower'], color='green', linestyle='--', label='Lower Threshold')
        axes[2].set_title('RSI')
        axes[2].legend()
        
        # Plot MACD
        axes[3].plot(indicators['macd'].index, indicators['macd'], label='MACD Histogram', color='blue')
        axes[3].axhline(y=0, color='black', linestyle='-')
        axes[3].set_title('MACD')
        axes[3].legend()
        
        # Plot Stochastic
        axes[4].plot(indicators['stoch'].index, indicators['stoch'], label='Stochastic', color='orange')
        axes[4].axhline(y=self.thresholds['stoch_upper'], color='red', linestyle='--', label='Upper Threshold')
        axes[4].axhline(y=self.thresholds['stoch_lower'], color='green', linestyle='--', label='Lower Threshold')
        axes[4].set_title('Stochastic Oscillator')
        axes[4].legend()
        
        # Plot CCI
        axes[5].plot(indicators['cci'].index, indicators['cci'], label='CCI', color='green')
        axes[5].axhline(y=self.thresholds['cci_upper'], color='red', linestyle='--', label='Upper Threshold')
        axes[5].axhline(y=self.thresholds['cci_lower'], color='green', linestyle='--', label='Lower Threshold')
        axes[5].set_title('CCI')
        axes[5].legend()
        
        # Add trade markers
        for ax in axes:
            for i in range(len(trades)):
                if trades.iloc[i, 0] > 0:  # Buy signal
                    ax.axvline(x=trades.index[i], color='green', alpha=0.3)
                elif trades.iloc[i, 0] < 0:  # Sell signal
                    ax.axvline(x=trades.index[i], color='red', alpha=0.3)
        
        plt.tight_layout()
        
        # Save or show the plot
        if output_file:
            output_path = get_output_path()
            plt.savefig(os.path.join(output_path, output_file))
        else:
            plt.show()
        
        plt.close()