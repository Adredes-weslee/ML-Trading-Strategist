"""
Technical Indicators Module

This module provides implementations of various technical indicators
for financial analysis and algorithmic trading strategies.
"""

import pandas as pd
import numpy as np


class Indicators:
    """
    Technical indicators implementation for trading strategies.
    
    This class provides a comprehensive set of technical indicators
    commonly used in financial analysis and algorithmic trading.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Indicators class.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration parameters for indicators
        """
        self.config = config if config is not None else {}
    
    def sma_indicator(self, prices, window=20):
        """
        Calculate Simple Moving Average (SMA).
        
        Parameters:
        -----------
        prices : pandas.Series
            Price data
        window : int, optional
            Window size, default 20
            
        Returns:
        --------
        pandas.Series
            SMA values
        """
        return prices.rolling(window=window).mean()
    
    def ema_indicator(self, prices, window=20):
        """
        Calculate Exponential Moving Average (EMA).
        
        Parameters:
        -----------
        prices : pandas.Series
            Price data
        window : int, optional
            Window size, default 20
            
        Returns:
        --------
        pandas.Series
            EMA values
        """
        return prices.ewm(span=window).mean()
    
    def bollinger_indicator(self, prices, window=20, num_std=2):
        """
        Calculate Bollinger Bands percentage (price relative to bands width).
        
        Parameters:
        -----------
        prices : pandas.Series
            Price data
        window : int, optional
            Window size for SMA, default 20
        num_std : int, optional
            Number of standard deviations for bands, default 2
            
        Returns:
        --------
        pandas.Series
            Bollinger Band percentage indicator
        """
        # Calculate SMA and standard deviation
        sma = self.sma_indicator(prices, window)
        rolling_std = prices.rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = sma + (rolling_std * num_std)
        lower_band = sma - (rolling_std * num_std)
        
        # Calculate (price - sma) / (upper_band - lower_band)
        bb_pct = (prices - sma) / (upper_band - lower_band)
        
        return bb_pct
    
    def bollinger_percent_indicator(self, prices, window=20, num_std=2):
        """
        Calculate Bollinger Bands percentage (price relative to bands width).
        
        Parameters:
        -----------
        prices : pandas.Series
            Price data
        window : int, optional
            Window size for SMA, default 20
        num_std : int, optional
            Number of standard deviations for bands, default 2
            
        Returns:
        --------
        pandas.Series
            Bollinger Band percentage indicator
        """
        # Just call bollinger_indicator for compatibility
        return self.bollinger_indicator(prices, window, num_std)
    
    def macd_indicator(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Parameters:
        -----------
        prices : pandas.Series
            Price data
        fast_period : int, optional
            Fast EMA period, default 12
        slow_period : int, optional
            Slow EMA period, default 26
        signal_period : int, optional
            Signal line EMA period, default 9
            
        Returns:
        --------
        tuple
            (MACD line, signal line)
        """
        # Calculate fast and slow EMAs
        fast_ema = self.ema_indicator(prices, fast_period)
        slow_ema = self.ema_indicator(prices, slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD line)
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        return macd_line, signal_line
    
    def rsi_indicator(self, prices, window=14):
        """
        Calculate Relative Strength Index (RSI).
        
        Parameters:
        -----------
        prices : pandas.Series
            Price data
        window : int, optional
            Window size, default 14
            
        Returns:
        --------
        pandas.Series
            RSI values
        """
        # Calculate price changes
        price_diff = prices.diff()
        
        # Create gains (upward) and losses (downward) Series
        gains = price_diff.copy()
        gains[gains < 0] = 0.0
        losses = -price_diff.copy()
        losses[losses < 0] = 0.0
        
        # Calculate average gains and losses
        avg_gain = gains.rolling(window=window).mean()
        avg_loss = losses.rolling(window=window).mean()
        
        # Calculate RS (Relative Strength)
        # Fix the deprecated method: Using replace with a fixed value instead of method='ffill'
        # And handle division by zero by replacing zeros with small value
        avg_loss_non_zero = avg_loss.copy()
        avg_loss_non_zero = avg_loss_non_zero.replace(0, 1e-9)
        rs = avg_gain / avg_loss_non_zero
        
        # Calculate RSI
        rsi_val = 100.0 - (100.0 / (1.0 + rs))
        
        # Handle NaN and edge cases
        # Fix the deprecated fillna(method='bfill') method
        return rsi_val.bfill().fillna(50.0)
    
    def stoch_indicator(self, prices, window=14, k_period=3, d_period=3):
        """
        Calculate Stochastic Oscillator.
        
        Parameters:
        -----------
        prices : pandas.Series
            Price data
        window : int, optional
            Window size, default 14
        k_period : int, optional
            %K smoothing period, default 3
        d_period : int, optional
            %D smoothing period, default 3
            
        Returns:
        --------
        pandas.Series
            Stochastic oscillator values
        """
        # Find highest high and lowest low in the window
        highest_high = prices.rolling(window=window).max()
        lowest_low = prices.rolling(window=window).min()
        
        # Calculate %K
        k = 100 * ((prices - lowest_low) / (highest_high - lowest_low))
        
        # Smooth %K (optional)
        if k_period > 1:
            k = k.rolling(window=k_period).mean()
        
        # Calculate %D (moving average of %K)
        d = k.rolling(window=d_period).mean()
        
        return k  # Return %K (common choice for trading signals)
    
    def atr_indicator(self, prices_high, prices_low, prices_close, window=14):
        """
        Calculate Average True Range (ATR).
        
        Parameters:
        -----------
        prices_high : pandas.Series
            High price data
        prices_low : pandas.Series
            Low price data
        prices_close : pandas.Series
            Close price data
        window : int, optional
            Window size, default 14
            
        Returns:
        --------
        pandas.Series
            ATR values
        """
        # Get previous close
        prev_close = prices_close.shift(1)
        
        # Calculate the three differences
        tr1 = prices_high - prices_low  # High - Low
        tr2 = abs(prices_high - prev_close)  # |High - Previous Close|
        tr3 = abs(prices_low - prev_close)  # |Low - Previous Close|
        
        # True Range is the maximum of the three
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Calculate ATR as simple moving average of True Range
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def cci_indicator(self, prices, window=20):
        """
        Calculate Commodity Channel Index (CCI).
        
        Parameters:
        -----------
        prices : pandas.Series
            Price data
        window : int, optional
            Window size, default 20
            
        Returns:
        --------
        pandas.Series
            CCI values
        """
        # Calculate typical price
        typical_price = prices
        
        # Calculate simple moving average of typical price
        sma_tp = self.sma_indicator(typical_price, window)
        
        # Calculate mean deviation
        mean_deviation = typical_price.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean())
        
        # Calculate CCI
        # Constant 0.015 is by definition of CCI
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    def momentum_indicator(self, prices, period=10):
        """
        Calculate Momentum.
        
        Parameters:
        -----------
        prices : pandas.Series
            Price data
        period : int, optional
            Period for momentum calculation, default 10
            
        Returns:
        --------
        pandas.Series
            Momentum values
        """
        # Momentum is current price - price 'period' days ago
        return prices - prices.shift(period)
    
    def obv_indicator(self, prices, volume):
        """
        Calculate On-Balance Volume (OBV).
        
        Parameters:
        -----------
        prices : pandas.Series
            Price data
        volume : pandas.Series
            Volume data
            
        Returns:
        --------
        pandas.Series
            OBV values
        """
        # Calculate price changes
        price_diff = prices.diff()
        
        # Create direction series
        direction = pd.Series(0, index=price_diff.index)
        direction[price_diff > 0] = 1
        direction[price_diff < 0] = -1
        
        # Calculate OBV
        obv = (direction * volume).cumsum()
        
        return obv