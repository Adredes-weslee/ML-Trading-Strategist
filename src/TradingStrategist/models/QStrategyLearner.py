"""
Q-Learning Strategy Learner

A trading strategy implementation that uses reinforcement learning (Q-Learning) to
make trading decisions based on market states derived from technical indicators.
"""

import datetime as dt
import pandas as pd
import numpy as np
from .QLearner import QLearner
from ..data.loader import get_data
from ..indicators.technical import Indicators


class QStrategyLearner:
    """
    A Reinforcement Learning based trading strategy using Q-Learning.
    
    This strategy discretizes market states based on technical indicators
    and learns optimal trading actions through rewards from portfolio performance.
    """
    
    def __init__(self, verbose=False, impact=0.0, commission=9.95,
                 # Technical parameters
                 indicator_bins=10,
                 window_size=20,
                 rsi_window=14,
                 stoch_window=14,
                 cci_window=20,
                 position_size=1000,
                 max_iterations=100,
                 # Indicator ranges
                 bb_range=(-2.0, 2.0),
                 rsi_range=(0.0, 1.0),
                 macd_range=(-1.0, 1.0),
                 stoch_range=(0.0, 1.0),
                 cci_range=(-2.0, 2.0),
                 # Normalization factors
                 rsi_norm=100.0,
                 stoch_norm=100.0,
                 cci_norm=200.0,
                 # Indicator selection flags
                 use_bb=True,
                 use_rsi=True,
                 use_macd=True,
                 use_stoch=False,
                 use_cci=False,
                 # Momentum parameters
                 momentum_periods=None,
                 # Convergence parameters
                 min_iterations=20,
                 convergence_threshold=0.1,
                 # Q-Learning parameters
                 learning_rate=0.2,
                 discount_factor=0.9,
                 random_action_rate=0.5,
                 random_action_decay=0.99,
                 dyna_iterations=10,
                 indicators_config=None):
        """
        Initialize the Q-Strategy Learner.
        
        Parameters:
        -----------
        verbose : bool, optional
            Whether to output additional information, default False
        impact : float, optional
            Market impact of trades, default 0.0
        commission : float, optional
            Commission cost of trades, default 9.95
        indicator_bins : int, optional
            Number of bins for discretizing indicators, default 10
        window_size : int, optional
            Window size for technical indicators, default 20
        rsi_window : int, optional
            Window size for RSI calculation, default 14
        stoch_window : int, optional
            Window size for Stochastic oscillator, default 14
        cci_window : int, optional
            Window size for CCI calculation, default 20
        position_size : int, optional
            Number of shares for positions, default 1000
        max_iterations : int, optional
            Maximum number of training iterations, default 100
        bb_range : tuple, optional
            Range for Bollinger bands discretization, default (-2.0, 2.0)
        rsi_range : tuple, optional
            Range for RSI discretization, default (0.0, 1.0)
        macd_range : tuple, optional
            Range for MACD discretization, default (-1.0, 1.0)
        stoch_range : tuple, optional
            Range for Stochastic oscillator discretization, default (0.0, 1.0)
        cci_range : tuple, optional
            Range for CCI discretization, default (-2.0, 2.0)
        rsi_norm : float, optional
            Normalization factor for RSI, default 100.0
        stoch_norm : float, optional
            Normalization factor for Stochastic Oscillator, default 100.0
        cci_norm : float, optional
            Normalization factor for CCI, default 200.0
        use_bb : bool, optional
            Whether to use Bollinger Bands indicator, default True
        use_rsi : bool, optional
            Whether to use RSI indicator, default True
        use_macd : bool, optional
            Whether to use MACD indicator, default True
        use_stoch : bool, optional
            Whether to use Stochastic oscillator, default False
        use_cci : bool, optional
            Whether to use CCI indicator, default False
        momentum_periods : list, optional
            List of periods for momentum calculations, default None
        min_iterations : int, optional
            Minimum iterations before checking convergence, default 20
        convergence_threshold : float, optional
            Threshold for convergence detection, default 0.1
        learning_rate : float, optional
            Learning rate (alpha) for Q-learning, default 0.2
        discount_factor : float, optional
            Discount factor (gamma) for future rewards, default 0.9
        random_action_rate : float, optional
            Initial random action rate, default 0.5
        random_action_decay : float, optional
            Decay rate for random actions, default 0.99
        dyna_iterations : int, optional
            Number of Dyna-Q planning updates per step, default 10
        indicators_config : dict, optional
            Configuration dictionary for technical indicators
        """
        # Input validation
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean")
        if not isinstance(impact, (int, float)) or impact < 0:
            raise ValueError("impact must be a non-negative number")
        if not isinstance(commission, (int, float)) or commission < 0:
            raise ValueError("commission must be a non-negative number")
        if not isinstance(indicator_bins, int) or indicator_bins <= 0:
            raise ValueError("indicator_bins must be a positive integer")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if not isinstance(rsi_window, int) or rsi_window <= 0:
            raise ValueError("rsi_window must be a positive integer")
        if not isinstance(stoch_window, int) or stoch_window <= 0:
            raise ValueError("stoch_window must be a positive integer")
        if not isinstance(cci_window, int) or cci_window <= 0:
            raise ValueError("cci_window must be a positive integer")
        if not isinstance(position_size, int) or position_size <= 0:
            raise ValueError("position_size must be a positive integer")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            raise ValueError("max_iterations must be a positive integer")
        
        # Initialize basic parameters
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.learner = None
        
        # Initialize indicators with config
        self.indicators = Indicators(config=indicators_config)
        
        # Store all configurable parameters
        self.indicator_bins = indicator_bins
        self.window_size = window_size
        self.rsi_window = rsi_window
        self.stoch_window = stoch_window
        self.cci_window = cci_window
        self.position_size = position_size
        self.max_iterations = max_iterations
        
        # Store range parameters
        self.bb_range = bb_range
        self.rsi_range = rsi_range
        self.macd_range = macd_range
        self.stoch_range = stoch_range
        self.cci_range = cci_range
        
        # Store normalization factors
        self.rsi_norm = rsi_norm
        self.stoch_norm = stoch_norm
        self.cci_norm = cci_norm
        
        # Store indicator selection flags
        self.use_bb = use_bb
        self.use_rsi = use_rsi
        self.use_macd = use_macd
        self.use_stoch = use_stoch
        self.use_cci = use_cci
        
        # Validate that at least one indicator is selected
        if not (use_bb or use_rsi or use_macd or use_stoch or use_cci):
            raise ValueError("At least one indicator must be selected")
            
        # Store momentum parameters
        self.momentum_periods = momentum_periods if momentum_periods is not None else []
        
        # Store convergence parameters
        self.min_iterations = min_iterations
        self.convergence_threshold = convergence_threshold
        
        # Q-Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.random_action_rate = random_action_rate
        self.random_action_decay = random_action_decay
        self.dyna_iterations = dyna_iterations
        
        # Trading positions
        self.positions = {
            0: 0,                   # No position
            1: position_size,       # Long position
            2: -position_size       # Short position
        }
        
    def _discretize(self, value, min_val, max_val, bins):
        """
        Discretize a continuous value into a discrete bin.
        
        Parameters:
        -----------
        value : float
            Value to discretize
        min_val : float
            Minimum value in range
        max_val : float
            Maximum value in range
        bins : int
            Number of bins
            
        Returns:
        --------
        int
            Bin index (0 to bins-1)
        """
        # Handle NaN values
        if np.isnan(value):
            return 0  # Default to first bin for NaN values
            
        if value <= min_val:
            return 0
        if value >= max_val:
            return bins - 1
            
        bin_width = (max_val - min_val) / bins
        return int((value - min_val) / bin_width)
        
    def _calculate_state(self, indicators, timestamp):
        """
        Calculate the state index from technical indicators.
        
        Parameters:
        -----------
        indicators : dict
            Dictionary of technical indicators
        timestamp : datetime
            Current timestamp
            
        Returns:
        --------
        int
            State index
        """
        # Get indicator values for this timestamp
        state_components = []
        
        # Add only the selected indicators
        if self.use_bb and 'bollinger' in indicators:
            bb_value = indicators['bollinger'].loc[timestamp]
            bb_bin = self._discretize(bb_value, self.bb_range[0], self.bb_range[1], self.indicator_bins)
            state_components.append(bb_bin)
            
        if self.use_rsi and 'rsi' in indicators:
            rsi_value = indicators['rsi'].loc[timestamp] / self.rsi_norm
            rsi_bin = self._discretize(rsi_value, self.rsi_range[0], self.rsi_range[1], self.indicator_bins)
            state_components.append(rsi_bin)
            
        if self.use_macd and 'macd' in indicators:
            macd_value = indicators['macd'].loc[timestamp]
            macd_bin = self._discretize(macd_value, self.macd_range[0], self.macd_range[1], self.indicator_bins)
            state_components.append(macd_bin)
            
        if self.use_stoch and 'stoch' in indicators:
            stoch_value = indicators['stoch'].loc[timestamp] / self.stoch_norm
            stoch_bin = self._discretize(stoch_value, self.stoch_range[0], self.stoch_range[1], self.indicator_bins)
            state_components.append(stoch_bin)
            
        if self.use_cci and 'cci' in indicators:
            cci_value = indicators['cci'].loc[timestamp] / self.cci_norm
            cci_bin = self._discretize(cci_value, self.cci_range[0], self.cci_range[1], self.indicator_bins)
            state_components.append(cci_bin)
        
        # Add momentum indicators if specified
        for i, period in enumerate(self.momentum_periods):
            if f'momentum_{period}' in indicators:
                momentum_value = indicators[f'momentum_{period}'].loc[timestamp]
                momentum_bin = self._discretize(momentum_value, -0.1, 0.1, self.indicator_bins)
                state_components.append(momentum_bin)
        
        # Combine the state components into a single state index
        if not state_components:
            return 0  # Default state if no components are available
            
        # Calculate the state index by treating the components as a base-N number
        state = 0
        for i, component in enumerate(reversed(state_components)):
            state += component * (self.indicator_bins ** i)
            
        return state
        
    def _calculate_reward(self, daily_returns, action, prev_action):
        """
        Calculate the reward for a given action based on returns.
        
        Parameters:
        -----------
        daily_returns : float
            Daily return for the current day
        action : int
            Action taken (0: no position, 1: long, 2: short)
        prev_action : int
            Previous action
            
        Returns:
        --------
        float
            Calculated reward
        """
        # Position-based reward
        position_reward = 0.0
        
        if action == 1:  # Long position
            position_reward = daily_returns
        elif action == 2:  # Short position
            position_reward = -daily_returns
            
        # Transaction cost penalty
        transaction_cost = 0.0
        if action != prev_action:
            # Add penalty for switching positions based on impact and commission
            transaction_cost = self.commission + abs(self.positions[action] - self.positions[prev_action]) * self.impact
            
        return position_reward - transaction_cost
    
    def _compute_indicators(self, prices, symbol):
        """
        Compute technical indicators for the given price data.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        symbol : str
            Stock symbol
            
        Returns:
        --------
        dict
            Dictionary of computed indicators
        """
        price_series = prices[symbol]
        indicators = {'daily_returns': price_series.pct_change()}
        
        # Compute selected indicators
        if self.use_bb:
            indicators['bollinger'] = self.indicators.bollinger_indicator(price_series, window=self.window_size)
            
        if self.use_rsi:
            indicators['rsi'] = self.indicators.rsi_indicator(price_series, window=self.rsi_window)
            
        if self.use_macd:
            # Get MACD and properly extract the histogram
            macd_result = self.indicators.macd_indicator(price_series)
            # Check if the result is a tuple (standard implementation) or already the histogram
            if isinstance(macd_result, tuple):
                macd_line, signal_line = macd_result
                macd_hist = macd_line - signal_line  # MACD histogram is the difference
            else:
                # If already processed as a single value
                macd_hist = macd_result
            indicators['macd'] = macd_hist
            
        if self.use_stoch:
            indicators['stoch'] = self.indicators.stoch_indicator(price_series, window=self.stoch_window)
            
        if self.use_cci:
            indicators['cci'] = self.indicators.cci_indicator(price_series, window=self.cci_window)
            
        # Add momentum indicators if specified
        for period in self.momentum_periods:
            indicators[f'momentum_{period}'] = price_series.pct_change(period)
            
        return indicators
        
    def addEvidence(self, symbol="JPM", sd=dt.datetime(2008, 1, 1), 
                   ed=dt.datetime(2009, 12, 31), sv=10000):
        """
        Train the Q-Learning strategy using price data.
        
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
        # Input validation for parameters
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("Symbol must be a non-empty string")
        if not isinstance(sd, dt.datetime) or not isinstance(ed, dt.datetime):
            raise ValueError("Start and end dates must be datetime objects")
        if ed <= sd:
            raise ValueError("End date must be after start date")
        if not isinstance(sv, (int, float)) or sv <= 0:
            raise ValueError("Starting value must be a positive number")
            
        # Get price data
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates, addSPY=True)
        prices = prices_all[[symbol]].copy()
        
        if self.verbose:
            print(f"Training Q-Learner on {len(prices)} days of {symbol} data")
        
        # Compute indicators
        indicators = self._compute_indicators(prices_all, symbol)
        daily_returns = indicators['daily_returns']
        
        # Calculate number of active indicators
        active_indicators_count = sum([
            1 if self.use_bb else 0,
            1 if self.use_rsi else 0,
            1 if self.use_macd else 0,
            1 if self.use_stoch else 0,
            1 if self.use_cci else 0
        ]) + len(self.momentum_periods)
        
        if active_indicators_count == 0:
            raise ValueError("At least one indicator must be selected")
        
        # Calculate total number of possible states
        num_states = self.indicator_bins ** active_indicators_count
        
        if self.verbose:
            print(f"Using {active_indicators_count} indicators with {self.indicator_bins} bins each")
            print(f"Total possible states: {num_states}")
        
        # Initialize Q-Learner with parameters from config
        self.learner = QLearner(
            num_states=num_states,
            num_actions=3,  # 0: no position, 1: long, 2: short
            alpha=self.learning_rate,
            gamma=self.discount_factor,
            rar=self.random_action_rate,
            radr=self.random_action_decay,
            dyna=self.dyna_iterations,
            verbose=self.verbose
        )
        
        # Prepare for training
        timestamps = prices.index.tolist()
        converged = False
        iteration = 0
        
        # Training loop
        while not converged and iteration < self.max_iterations:
            cumulative_reward = 0.0
            prev_action = 0  # Start with no position
            
            # Skip the first day since we don't have returns
            for i in range(1, len(timestamps)):
                date = timestamps[i]
                
                # Get state for current day
                state = self._calculate_state(indicators, date)
                
                # Calculate reward for previous action
                if i > 1:
                    reward = self._calculate_reward(
                        daily_returns.loc[date], 
                        prev_action, 
                        prev_action if i == 1 else prev_prev_action
                    )
                    cumulative_reward += reward
                else:
                    reward = 0.0
                
                # Get new action from Q-learner
                if i == 1:
                    action = self.learner.querysetstate(state)
                else:
                    action = self.learner.query(state, reward)
                
                prev_prev_action = prev_action
                prev_action = action
            
            # Check for convergence (stabilized cumulative reward)
            iteration += 1
            if self.verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}, Cumulative Reward: {cumulative_reward}")
                
            # Use configurable convergence parameters
            if iteration > self.min_iterations and abs(cumulative_reward) < self.convergence_threshold:
                converged = True
        
        if self.verbose:
            print(f"Training completed after {iteration} iterations")
    
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
        prices = prices_all[[symbol]].copy()
        
        if self.verbose:
            print(f"Testing on {len(prices)} days of {symbol} data")
        
        # Compute indicators
        indicators = self._compute_indicators(prices_all, symbol)
        
        # Initialize trades DataFrame
        trades = pd.DataFrame(0, index=prices.index, columns=[symbol])
        
        # Initial position
        current_position = 0  # 0: no position, 1: long, 2: short
        timestamps = prices.index.tolist()
        
        # Go through each day
        for i, date in enumerate(timestamps):
            # Calculate state
            state = self._calculate_state(indicators, date)
            
            # Get action from Q-learner (no learning during testing)
            if i == 0:
                action = self.learner.querysetstate(state)
            else:
                # Dummy reward during testing (not used for learning)
                action = self.learner.querysetstate(state)
            
            # Convert action to trades
            new_position = action
            
            # Only trade if position changed
            if new_position != current_position:
                # Calculate change in shares
                position_change = self.positions[new_position] - self.positions[current_position]
                trades.loc[date, symbol] = position_change
                
                current_position = new_position
        
        return trades
        
    def get_used_indicators(self):
        """
        Get a list of indicators currently being used by the model.
        
        Returns:
        --------
        list
            List of indicator names being used
        """
        indicators = []
        if self.use_bb:
            indicators.append("Bollinger Bands")
        if self.use_rsi:
            indicators.append("RSI")
        if self.use_macd:
            indicators.append("MACD")
        if self.use_stoch:
            indicators.append("Stochastic Oscillator")
        if self.use_cci:
            indicators.append("CCI")
        for period in self.momentum_periods:
            indicators.append(f"Momentum ({period} days)")
        return indicators