"""
Q-Learning Implementation

This module implements a Q-Learning algorithm for reinforcement learning problems.
Q-Learning finds an optimal action-selection policy using a reward maximization approach.
"""

import random
import numpy as np


class QLearner:
    """
    A tabular Q-learning implementation.
    
    Q-Learning is a model-free reinforcement learning algorithm that learns
    the value of actions in states by updating a Q-table based on rewards.
    This implementation includes optional Dyna-Q for planning.
    """

    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False,
    ):
        """
        Initialize the Q-Learner.
        
        Parameters:
        -----------
        num_states : int, optional
            Number of states in the environment, default 100
        num_actions : int, optional
            Number of actions available, default 4
        alpha : float, optional
            Learning rate (0-1), default 0.2
        gamma : float, optional
            Discount factor for future rewards (0-1), default 0.9
        rar : float, optional
            Random action rate (0-1), default 0.5
        radr : float, optional
            Random action decay rate (0-1), default 0.99
        dyna : int, optional
            Number of Dyna-Q planning updates per step, default 0
        verbose : bool, optional
            Whether to print debugging information, default False
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna

        # Initialize Q-table and model for Dyna-Q
        self.Q = np.zeros((num_states, num_actions))
        self.s = 0  # Current state
        self.a = 0  # Current action
        self.model = {}  # For Dyna-Q planning

    def querysetstate(self, s):
        """
        Update the current state without updating the Q-table.
        
        Parameters:
        -----------
        s : int
            The new state
            
        Returns:
        --------
        int
            The selected action
        """
        self.s = s
        
        # Select action using epsilon-greedy policy
        if random.random() < self.rar:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s, :])
            
        self.a = action
        
        if self.verbose:
            print(f"querysetstate() => s = {s}, a = {action}")
            
        return action

    def query(self, s_prime, r):
        """
        Update the Q table and return an action.
        
        Parameters:
        -----------
        s_prime : int
            The new state
        r : float
            The immediate reward
            
        Returns:
        --------
        int
            The selected action
        """
        # Q-Learning update rule
        old_q = self.Q[self.s, self.a]
        max_q_next = np.max(self.Q[s_prime, :])
        self.Q[self.s, self.a] = (1 - self.alpha) * old_q + \
            self.alpha * (r + self.gamma * max_q_next)

        # Store the experience in the model for Dyna-Q
        self.model[(self.s, self.a)] = (s_prime, r)

        # Select action for the new state using epsilon-greedy
        if random.random() < self.rar:
            action = random.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s_prime, :])

        # Decay random action rate
        self.rar *= self.radr

        # Perform Dyna-Q planning updates if enabled
        for _ in range(self.dyna):
            # Randomly select a previously observed state-action pair
            s_rand, a_rand = random.choice(list(self.model.keys()))
            s_rand_prime, r_rand = self.model[(s_rand, a_rand)]
            
            # Update Q-value using the same update rule
            old_q_dyna = self.Q[s_rand, a_rand]
            max_q_next_dyna = np.max(self.Q[s_rand_prime, :])
            self.Q[s_rand, a_rand] = (1 - self.alpha) * old_q_dyna + \
                self.alpha * (r_rand + self.gamma * max_q_next_dyna)

        # Update current state and action
        self.s = s_prime
        self.a = action
        
        if self.verbose:
            print(f"query() => s = {s_prime}, a = {action}, r = {r}")
            
        return action