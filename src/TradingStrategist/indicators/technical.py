"""
Technical Indicators for Trading Strategy Analysis

This module provides implementations of common technical indicators used
in trading strategies, including Bollinger Bands, RSI, MACD, Stochastic
Oscillator, and CCI.
"""

import pandas as pd
import numpy as np


class Indicators:
    """
    A collection of technical indicators for market analysis.
    
    Each indicator method accepts price data and returns the calculated
    indicator values as a pandas Series.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Indicators class with optional configuration.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary for technical indicators
        """
        # Default configuration values
        self.config = {
            # Bollinger Bands
            'bb_window': 20,
            'bb_std_dev': 2.0,
            'bb_upper_threshold': 1.0,
            'bb_lower_threshold': -1.0,
            
            # RSI
            'rsi_window': 14,
            'rsi_upper_threshold': 70,
            'rsi_lower_threshold': 30,
            
            # MACD
            'macd_fast_period': 12,
            'macd_slow_period': 26,
            'macd_signal_period': 9,
            
            # Stochastic Oscillator
            'stoch_window': 14,
            'stoch_upper_threshold': 80,
            'stoch_lower_threshold': 20,
            
            # CCI
            'cci_window': 20,
            'cci_constant': 0.015,
            'cci_upper_threshold': 100,
            'cci_lower_threshold': -100
        }
        
        # Update with custom configuration if provided
        if config is not None:
            self.config.update(config)

    def bollinger_indicator(self, prices, window=None):
        """
        Calculate Bollinger Bands indicator.
        
        Parameters:
        -----------
        prices : pd.Series
            Price data
        window : int, optional
            Rolling window size, defaults to config value
            
        Returns:
        --------
        pd.Series
            Normalized Bollinger Bands values
        """
        window = window if window is not None else self.config['bb_window']
        std_dev = self.config['bb_std_dev']
        upper_threshold = self.config['bb_upper_threshold']
        lower_threshold = self.config['bb_lower_threshold']
        
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        bb = (prices - sma) / (std_dev * std)
        
        prev_bb = bb.shift(1)
        signals = pd.Series(0, index=bb.index)

        cross_down = (prev_bb >= lower_threshold) & (bb < lower_threshold)
        signals.loc[cross_down] = 1  

        cross_up = (prev_bb <= upper_threshold) & (bb > upper_threshold)
        signals.loc[cross_up] = -1  
        
        bb_line = (prices - sma) / (std_dev * std.replace(0, np.nan))
        return bb_line.fillna(0.0)

    def rsi_indicator(self, prices, window=None):
        """
        Calculate Relative Strength Index (RSI).
        
        Parameters:
        -----------
        prices : pd.Series
            Price data
        window : int, optional
            Rolling window size, defaults to config value
            
        Returns:
        --------
        pd.Series
            RSI values (0-100 scale)
        """
        window = window if window is not None else self.config['rsi_window']
        upper_threshold = self.config['rsi_upper_threshold']
        lower_threshold = self.config['rsi_lower_threshold']
        
        delta = prices.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.rolling(window=window, min_periods=window).mean()
        avg_loss = losses.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss.replace(to_replace=0, method='ffill').replace(np.nan, 1e-9)
        rsi_val = 100 - (100 / (1 + rs))

        prev_rsi = rsi_val.shift(1)
        signals = pd.Series(0, index=rsi_val.index)

        cross_down = (prev_rsi >= lower_threshold) & (rsi_val < lower_threshold)
        signals.loc[cross_down] = 1  

        cross_up = (prev_rsi <= upper_threshold) & (rsi_val > upper_threshold)
        signals.loc[cross_up] = -1  
        
        return rsi_val.fillna(method='bfill').fillna(50.0)

    def macd_indicator(self, prices, n_fast=None, n_slow=None, n_signal=None):
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Parameters:
        -----------
        prices : pd.Series
            Price data
        n_fast : int, optional
            Fast EMA period, defaults to config value
        n_slow : int, optional
            Slow EMA period, defaults to config value
        n_signal : int, optional
            Signal line period, defaults to config value
            
        Returns:
        --------
        pd.Series
            MACD histogram values
        """
        n_fast = n_fast if n_fast is not None else self.config['macd_fast_period']
        n_slow = n_slow if n_slow is not None else self.config['macd_slow_period']
        n_signal = n_signal if n_signal is not None else self.config['macd_signal_period']
        
        ema_fast = prices.ewm(span=n_fast, adjust=False).mean()
        ema_slow = prices.ewm(span=n_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=n_signal, adjust=False).mean()
        macd_hist = macd_line - signal_line

        prev_hist = macd_hist.shift(1)
        signals = pd.Series(0, index=macd_hist.index)

        cross_up = (prev_hist <= 0) & (macd_hist > 0)
        signals.loc[cross_up] = 1  

        cross_down = (prev_hist >= 0) & (macd_hist < 0)
        signals.loc[cross_down] = -1 
        
        return macd_hist.fillna(0.0)

    def stoch_indicator(self, prices, window=None):
        """
        Calculate Stochastic Oscillator.
        
        Parameters:
        -----------
        prices : pd.Series
            Price data
        window : int, optional
            Rolling window size, defaults to config value
            
        Returns:
        --------
        pd.Series
            Stochastic K values (0-100 scale)
        """
        window = window if window is not None else self.config['stoch_window']
        upper_threshold = self.config['stoch_upper_threshold']
        lower_threshold = self.config['stoch_lower_threshold']
        
        rolling_low = prices.rolling(window=window).min()
        rolling_high = prices.rolling(window=window).max()
        stoch_k = 100 * (prices - rolling_low) / (rolling_high - rolling_low + 1e-9)

        prev_stoch = stoch_k.shift(1)
        signals = pd.Series(0, index=stoch_k.index)

        cross_down = (prev_stoch >= lower_threshold) & (stoch_k < lower_threshold)
        signals.loc[cross_down] = 1  

        cross_up = (prev_stoch <= upper_threshold) & (stoch_k > upper_threshold)
        signals.loc[cross_up] = -1  
    
        return stoch_k.fillna(50.0)  

    def cci_indicator(self, prices, window=None):
        """
        Calculate Commodity Channel Index (CCI).
        
        Parameters:
        -----------
        prices : pd.Series
            Price data
        window : int, optional
            Rolling window size, defaults to config value
            
        Returns:
        --------
        pd.Series
            CCI values
        """
        window = window if window is not None else self.config['cci_window']
        constant = self.config['cci_constant']
        upper_threshold = self.config['cci_upper_threshold']
        lower_threshold = self.config['cci_lower_threshold']
        
        sma = prices.rolling(window=window).mean()
        mad = (prices - sma).abs().rolling(window=window).mean()
        cci = (prices - sma) / (constant * mad.replace(0, 1e-9))

        prev_cci = cci.shift(1)
        signals = pd.Series(0, index=cci.index)

        cross_down = (prev_cci >= lower_threshold) & (cci < lower_threshold)
        signals.loc[cross_down] = 1 

        cross_up = (prev_cci <= upper_threshold) & (cci > upper_threshold)
        signals.loc[cross_up] = -1 
        
        return cci.fillna(0.0)