"""
Tree Strategy Learner

A machine learning-based trading strategy that uses technical indicators
as features and a bagged random tree ensemble for learning and prediction.
"""

import datetime as dt
import pandas as pd
import numpy as np
from TradingStrategist.models.BagLearner import BagLearner
from TradingStrategist.models.RTLearner import RTLearner
from TradingStrategist.data.loader import get_data
from TradingStrategist.indicators.technical import Indicators


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
        macd = self.indicators.macd_indicator(price_series)
        stoch = self.indicators.stoch_indicator(price_series, window=self.stoch_window) / self.stoch_norm
        cci = self.indicators.cci_indicator(price_series, window=self.cci_window) / self.cci_norm
        
        # Combine indicators
        features = pd.DataFrame(index=prices.index)
        features['BB'] = bb
        features['RSI'] = rsi  
        features['MACD'] = macd
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
        # Get price data
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates, addSPY=True)
        prices = prices_all[[symbol]].copy()  # Just the symbol we're interested in
        
        if self.verbose:
            print(f"Training on {len(prices)} days of data")
        
        # Compute indicators for features
        features = self._compute_indicators(prices)
        
        # Generate labels for supervised learning
        labels = self._generate_labels(prices)
        
        # Remove rows with NaN values
        valid_indices = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_indices]
        labels = labels[valid_indices]
        
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