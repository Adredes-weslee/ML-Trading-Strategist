"""
Tree Strategy Learner

A machine learning-based trading strategy that uses technical indicators
as features and a bagged random tree ensemble for learning and prediction.
"""

import datetime as dt
import pandas as pd
import numpy as np
from .BagLearner import BagLearner
from .RTLearner import RTLearner
from ..data.loader import get_data
from ..indicators.technical import Indicators


class TreeStrategyLearner:
    """
    A Decision Tree-based strategy learner that uses ensemble methods
    (bagging with random trees) to make trading decisions.
    """
    
    def __init__(self, verbose=False, impact=0.0, commission=9.95, window_size=20,
             buy_threshold=0.02, sell_threshold=-0.02, prediction_days=5, 
             leaf_size=5, bags=20, boost=False, rsi_window=14, stoch_window=14,
             cci_window=20, position_size=1000, momentum_periods=None,
             rsi_norm=100.0, stoch_norm=100.0, cci_norm=200.0,
             indicators_config=None):
        """
        Initialize the TreeStrategyLearner.
        
        Parameters:
        -----------
        verbose : bool, optional
            Whether to output additional information, default False
        impact : float, optional
            Market impact of trades, default 0.0
        commission : float, optional
            Commission cost of trades, default 9.95
        window_size : int, optional
            Window size for indicators, default 20
        buy_threshold : float, optional
            Return threshold to generate buy signal, default 0.02
        sell_threshold : float, optional
            Return threshold to generate sell signal, default -0.02
        prediction_days : int, optional
            Number of days to look ahead for returns, default 5
        leaf_size : int, optional
            Leaf size for RT Learner, default 5
        bags : int, optional
            Number of bags for BagLearner, default 20
        boost : bool, optional
            Whether to use boosting, default False
        rsi_window : int, optional
            Window size for RSI calculation, default 14
        stoch_window : int, optional
            Window size for Stochastic Oscillator, default 14
        cci_window : int, optional
            Window size for CCI calculation, default 20
        position_size : int, optional
            Number of shares to trade, default 1000
        momentum_periods : list, optional
            List of periods for momentum calculations, default [3, 5, 10]
        rsi_norm : float, optional
            Normalization factor for RSI, default 100.0
        stoch_norm : float, optional
            Normalization factor for Stochastic Oscillator, default 100.0
        cci_norm : float, optional
            Normalization factor for CCI, default 200.0
        indicators_config : dict, optional
            Configuration dictionary for technical indicators
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        
        # Technical indicator parameters
        self.window_size = window_size
        self.rsi_window = rsi_window
        self.stoch_window = stoch_window
        self.cci_window = cci_window
        self.indicators = Indicators(config=indicators_config)
        
        # Normalization factors
        self.rsi_norm = rsi_norm
        self.stoch_norm = stoch_norm
        self.cci_norm = cci_norm
        
        # Momentum parameters
        self.momentum_periods = momentum_periods if momentum_periods is not None else [3, 5, 10]
        
        # Trading strategy parameters
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.prediction_days = prediction_days
        self.position_size = position_size
        
        # ML model parameters
        self.leaf_size = leaf_size
        self.bags = bags
        self.boost = boost
        
        self.learner = None

    def _compute_indicators(self, prices):
        """
        Compute technical indicators for feature generation.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data for a symbol
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing all computed indicators as features
        """
        price_series = prices.iloc[:, 0]  # Get the single price column
        
        # Use class parameters instead of hardcoded values
        bb = self.indicators.bollinger_indicator(price_series, window=self.window_size)
        rsi = self.indicators.rsi_indicator(price_series, window=self.rsi_window) / self.rsi_norm
        
        # Get MACD and properly extract the histogram
        macd_result = self.indicators.macd_indicator(price_series)
        # Check if the result is a tuple (standard implementation) or already the histogram
        if isinstance(macd_result, tuple):
            macd_line, signal_line = macd_result
            macd_hist = macd_line - signal_line  # MACD histogram is the difference
        else:
            # If already processed as a single value
            macd_hist = macd_result
        
        stoch = self.indicators.stoch_indicator(price_series, window=self.stoch_window) / self.stoch_norm
        cci = self.indicators.cci_indicator(price_series, window=self.cci_window) / self.cci_norm
        
        # Combine indicators
        features = pd.DataFrame(index=prices.index)
        features['BB'] = bb
        features['RSI'] = rsi  
        features['MACD'] = macd_hist  # Use the MACD histogram
        features['Stoch'] = stoch
        features['CCI'] = cci
        
        # Add price momentum features with configurable periods
        for days in self.momentum_periods:
            momentum = price_series.pct_change(days)
            features[f'Momentum_{days}'] = momentum
        
        # Ensure all features have finite values
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        return features
    
    def _generate_labels(self, prices):
        """
        Generate classification labels based on future returns.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data for a symbol
            
        Returns:
        --------
        pd.Series
            Labels for training: 1 (buy), -1 (sell), 0 (hold)
        """
        # Calculate future returns using the configurable prediction_days
        future_returns = prices.pct_change(self.prediction_days).shift(-self.prediction_days)
        
        # Initialize labels as hold (0)
        labels = pd.Series(0, index=prices.index)
        
        # Adjust for transaction costs
        adjusted_buy_threshold = self.buy_threshold + self.impact
        adjusted_sell_threshold = self.sell_threshold - self.impact
        
        # Set buy and sell signals based on future returns
        labels[future_returns.iloc[:, 0] >= adjusted_buy_threshold] = 1  # Buy
        labels[future_returns.iloc[:, 0] <= adjusted_sell_threshold] = -1  # Sell
        
        return labels
        
    def addEvidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), 
                   ed=dt.datetime(2009, 12, 31), sv=10000):
        """
        Train the strategy learner using price data.
        
        Parameters:
        -----------
        symbol : str, optional
            Stock symbol to train on, default 'JPM'
        sd : datetime, optional
            Start date for training data, default Jan 1, 2008
        ed : datetime, optional
            End date for training data, default Dec 31, 2009
        sv : int, optional
            Starting value of portfolio, default 10000
        """
        # Input validation
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("Symbol must be a non-empty string")
            
        if not isinstance(sd, dt.datetime) or not isinstance(ed, dt.datetime):
            raise ValueError("Start and end dates must be datetime objects")
            
        if ed <= sd:
            raise ValueError("End date must be after start date")
            
        if not isinstance(sv, (int, float)) or sv <= 0:
            raise ValueError("Starting value must be a positive number")
            
        # Parameter validation
        if self.window_size <= 0:
            raise ValueError("Window size must be positive")
            
        if self.rsi_window <= 0:
            raise ValueError("RSI window must be positive")
            
        if self.stoch_window <= 0:
            raise ValueError("Stochastic window must be positive")
            
        if self.cci_window <= 0:
            raise ValueError("CCI window must be positive")
            
        if self.leaf_size <= 0:
            raise ValueError("Leaf size must be positive")
            
        if self.bags <= 0:
            raise ValueError("Number of bags must be positive")
            
        if self.position_size <= 0:
            raise ValueError("Position size must be positive")
            
        if not all(period > 0 for period in self.momentum_periods):
            raise ValueError("Momentum periods must be positive")
            
        # Validate that momentum periods are sorted to avoid inconsistencies
        if sorted(self.momentum_periods) != self.momentum_periods:
            if self.verbose:
                print("Warning: Momentum periods were not in ascending order. Sorting automatically.")
            self.momentum_periods = sorted(self.momentum_periods)
            
        # Calculate minimum data length needed based on indicators
        min_data_length = max(
            self.window_size,
            self.rsi_window,
            self.stoch_window,
            self.cci_window,
            max(self.momentum_periods) if self.momentum_periods else 0,
        ) + self.prediction_days + 10  # Extra buffer
        
        # Get price data
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates, addSPY=True)
        prices = prices_all[[symbol]].copy()  # Just the symbol we're interested in
        
        if len(prices) < min_data_length:
            raise ValueError(f"Insufficient data: at least {min_data_length} data points required, but only {len(prices)} provided")
            
        if self.verbose:
            print(f"Training on {len(prices)} days of data")
        
        # Check for missing or invalid values in the price data
        if prices.isna().any().any():
            if self.verbose:
                print("Warning: Price data contains NaN values. These will be filled using forward-fill method.")
            prices = prices.ffill().bfill()
            if prices.isna().any().any():
                raise ValueError("Price data contains NaN values that couldn't be filled")
        
        # Compute indicators for features
        features = self._compute_indicators(prices)
        
        # Generate labels for supervised learning
        labels = self._generate_labels(prices)
        
        # Remove rows with NaN values
        valid_indices = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_indices]
        labels = labels[valid_indices]
        
        if len(features) < 10:  # Arbitrary minimum sample threshold
            raise ValueError(f"Too few valid training samples: {len(features)}. Check your data quality.")
        
        if self.verbose:
            print(f"Using {features.shape[1]} features and {len(features)} training samples")
        
        # Create and train the learner using configurable parameters
        self.learner = BagLearner(
            learner=RTLearner,
            kwargs={"leaf_size": self.leaf_size},
            bags=self.bags,
            boost=self.boost,
            verbose=self.verbose
        )
        
        self.learner.addEvidence(features.values, labels.values)
        
        if self.verbose:
            print("Training complete")
    
    def testPolicy(self, symbol="JPM", sd=dt.datetime(2010, 1, 1), 
                  ed=dt.datetime(2011, 12, 31), sv=10000):
        """
        Test the learned policy on price data and generate trades.
        
        Parameters:
        -----------
        symbol : str, optional
            Stock symbol to test on, default 'JPM'
        sd : datetime, optional
            Start date for testing data, default Jan 1, 2010
        ed : datetime, optional
            End date for testing data, default Dec 31, 2011
        sv : int, optional
            Starting value of portfolio, default 10000
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing trades: +position_size for buy, -position_size for sell, 0 for hold
        """
        if self.learner is None:
            raise ValueError("Model has not been trained. Call addEvidence() first.")
        
        # Get price data
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates, addSPY=True)
        prices = prices_all[[symbol]].copy()  # Just the symbol we're interested in
        
        if self.verbose:
            print(f"Testing on {len(prices)} days of data")
        
        # Compute indicators for features
        features = self._compute_indicators(prices)
        
        # Make predictions
        predictions = self.learner.query(features.values)
        
        # Convert predictions to trades
        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        
        # Keep track of position to avoid duplicating positions
        position = 0  # -1: short, 0: cash, 1: long
        
        for i in range(len(predictions)):
            date = prices.index[i]
            pred = predictions[i]
            
            # Long position signal
            if pred > 0 and position <= 0:
                trades.loc[date, symbol] = self.position_size  # Use configured position size
                position = 1
            
            # Short position signal
            elif pred < 0 and position >= 0:
                trades.loc[date, symbol] = -self.position_size  # Use configured position size
                position = -1
        
        if self.verbose:
            trade_count = (trades != 0).sum().sum()
            print(f"Generated {trade_count} trades")
        
        return trades
    
    def get_feature_importances(self):
        """
        Get feature importance scores from the tree model.
        
        Returns:
        --------
        dict
            Dictionary mapping feature names to importance scores
        """
        if self.learner is None:
            raise ValueError("Model has not been trained. Call addEvidence() first.")
            
        # Get feature importance from the BagLearner
        importances = self.learner.get_feature_importances() if hasattr(self.learner, 'get_feature_importances') else None
        
        if importances is not None:
            # Generate feature names dynamically based on momentum_periods
            feature_names = ['BB', 'RSI', 'MACD', 'Stoch', 'CCI']
            for period in self.momentum_periods:
                feature_names.append(f'Momentum_{period}')
            
            return {name: imp for name, imp in zip(feature_names, importances)}
        else:
            return None