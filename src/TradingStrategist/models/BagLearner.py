"""
Bag Learner

An ensemble learning implementation that combines multiple instances of a base learner
to improve robustness and reduce overfitting through bootstrap aggregation (bagging).
"""

import numpy as np


class BagLearner:
    """
    A bootstrap aggregation (bagging) ensemble learner.
    
    This class creates multiple instances of a base learning algorithm, trains each
    on a random bootstrap sample of the training data, and aggregates their predictions.
    """
    
    def __init__(self, learner, kwargs, bags=20, boost=False, verbose=False):
        """
        Initialize the BagLearner.
        
        Parameters:
        -----------
        learner : class
            The learning algorithm class to use as the base learner
        kwargs : dict
            Arguments to pass to the base learner
        bags : int, optional
            Number of learners to create, default 20
        boost : bool, optional
            Whether to use boosting (not implemented), default False
        verbose : bool, optional
            Whether to output additional information, default False
        """
        self.learner = learner
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost  # Note: Boosting not implemented yet
        self.verbose = verbose
        self.learners = []
        
        # Create the ensemble of learners
        for _ in range(self.bags):
            self.learners.append(learner(**kwargs))
    
    def addEvidence(self, data_x, data_y):
        """
        Train each learner in the ensemble on a bootstrap sample.
        
        Parameters:
        -----------
        data_x : numpy.ndarray
            Features/predictors for training
        data_y : numpy.ndarray
            Target values for training
        """
        # Check that data dimensions are compatible
        if len(data_x) != len(data_y):
            raise ValueError("data_x and data_y must have the same length")
            
        if self.verbose:
            print(f"Training {self.bags} learners with {len(data_x)} samples")
        
        # Train each learner on a bootstrap sample
        for i, learner in enumerate(self.learners):
            # Create bootstrap sample (sampling with replacement)
            sample_indices = np.random.choice(len(data_x), len(data_x), replace=True)
            sample_x = data_x[sample_indices]
            sample_y = data_y[sample_indices]
            
            # Train the learner
            learner.addEvidence(sample_x, sample_y)
            
            if self.verbose and (i + 1) % 5 == 0:
                print(f"Trained {i + 1}/{self.bags} learners")
    
    def query(self, points):
        """
        Make predictions by averaging the outputs of all learners.
        
        Parameters:
        -----------
        points : numpy.ndarray
            Feature data to make predictions for
            
        Returns:
        --------
        numpy.ndarray
            Aggregated predictions from all learners
        """
        # Get predictions from each learner
        predictions = np.array([learner.query(points) for learner in self.learners])
        
        # Aggregate predictions (mean for regression, mode for classification)
        result = np.mean(predictions, axis=0)
        
        return result
        
    def get_feature_importances(self):
        """
        Get the average feature importance across all learners in the ensemble.
        
        Returns:
        --------
        numpy.ndarray: Feature importance scores averaged across all base learners.
        """
        if not hasattr(self.learners[0], 'get_feature_importances'):
            raise AttributeError("Base learner does not support feature importance calculation")
        
        # Get feature importances from each learner and average them
        importances = [learner.get_feature_importances() for learner in self.learners]
        return np.mean(importances, axis=0)