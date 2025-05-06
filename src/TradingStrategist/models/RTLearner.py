"""
Random Tree Learner

A decision tree implementation that uses randomization when selecting split features
to improve robustness and reduce overfitting.
"""

import numpy as np


class RTLearner:
    """
    Random Tree Learner implementation.
    
    This class implements a decision tree that randomly selects features
    when determining splits, making it less prone to overfitting than
    standard deterministic trees.
    """
    
    def __init__(self, leaf_size=1, verbose=False):
        """
        Initialize the Random Tree Learner.
        
        Parameters:
        -----------
        leaf_size : int, optional
            Maximum number of samples to aggregate into a leaf, default 1
        verbose : bool, optional
            Whether to output additional information, default False
        """
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = None  # Will be built during training
        self.feature_count = None  # Set during training
    
    def addEvidence(self, data_x, data_y):
        """
        Build a random decision tree using the training data.
        
        Parameters:
        -----------
        data_x : numpy.ndarray
            Features/predictors for training
        data_y : numpy.ndarray
            Target values for training
        """
        # Check for valid inputs
        if len(data_x) != len(data_y):
            raise ValueError("data_x and data_y must have the same length")
            
        if len(data_x) == 0:
            raise ValueError("Cannot train on empty dataset")
        
        # Store the feature count for feature importance calculation
        self.feature_count = data_x.shape[1]
        
        # Build the tree recursively
        self.tree = self._build_tree(data_x, data_y)
        
        if self.verbose:
            print(f"Built tree with {len(self.tree)} nodes")
    
    def _build_tree(self, data_x, data_y):
        """
        Recursively build the random decision tree.
        
        Parameters:
        -----------
        data_x : numpy.ndarray
            Features for the current node
        data_y : numpy.ndarray
            Target values for the current node
            
        Returns:
        --------
        numpy.ndarray
            Decision tree as a 2D array where each row represents a node:
            [feature_index, split_value, left_node, right_node]
            Leaf nodes have feature_index = -1 and split_value = prediction
        """
        # Check if we should create a leaf node
        if len(data_y) <= self.leaf_size or len(np.unique(data_y)) == 1:
            # Return a leaf node (feature = -1, split = mean of y values)
            return np.array([[-1, np.mean(data_y), -1, -1]])
        
        # Improved check for identical feature values across all data points
        feature_variance = np.var(data_x, axis=0)
        if np.all(feature_variance < 1e-8):  # All features are essentially constant
            if self.verbose:
                print("Warning: All features have the same value. Creating leaf node.")
            return np.array([[-1, np.mean(data_y), -1, -1]])  # Leaf node
            
        # Choose a random feature to split on
        # Avoid features with zero variance (all values the same)
        valid_features = np.where(feature_variance > 1e-8)[0]
        if len(valid_features) == 0:
            # If all features have the same values, create a leaf node
            return np.array([[-1, np.mean(data_y), -1, -1]])
            
        # Choose from valid features only
        feature = np.random.choice(valid_features)
        
        # Find two random distinct values if possible, or use median
        unique_values = np.unique(data_x[:, feature])
        if len(unique_values) > 1:
            # Choose two random samples and use their mean as split value
            sample_indices = np.random.choice(len(unique_values), 2, replace=False)
            split_val = np.mean([unique_values[sample_indices[0]], unique_values[sample_indices[1]]])
        else:
            # Only one unique value, can't split on this feature
            # Return a leaf node
            return np.array([[-1, np.mean(data_y), -1, -1]])
        
        # Split the data
        left_mask = data_x[:, feature] <= split_val
        
        # Handle edge case where all data goes to one side
        if np.all(left_mask) or np.all(~left_mask):
            return np.array([[-1, np.mean(data_y), -1, -1]])  # Leaf node
            
        left_tree = self._build_tree(data_x[left_mask], data_y[left_mask])
        right_tree = self._build_tree(data_x[~left_mask], data_y[~left_mask])
        
        # Create root node
        # [feature, split_value, left_subtree_start, right_subtree_start]
        root = np.array([[feature, split_val, 1, len(left_tree) + 1]])
        
        # Combine into a single tree
        return np.vstack((root, left_tree, right_tree))
    
    def query(self, points):
        """
        Make predictions using the trained decision tree.
        
        Parameters:
        -----------
        points : numpy.ndarray
            Feature data to make predictions for
            
        Returns:
        --------
        numpy.ndarray
            Predictions for the input points
        """
        if self.tree is None:
            raise ValueError("Tree not built yet. Call addEvidence() first.")
        
        # Make a prediction for each point
        predictions = np.zeros(len(points))
        for i, point in enumerate(points):
            # Start at the root (index 0)
            node_idx = 0
            
            # Traverse the tree until reaching a leaf
            while self.tree[node_idx, 0] != -1:  # While not a leaf
                feature = int(self.tree[node_idx, 0])
                split_val = self.tree[node_idx, 1]
                
                # Determine which branch to follow
                if point[feature] <= split_val:
                    # Go left
                    node_idx += int(self.tree[node_idx, 2])
                else:
                    # Go right
                    node_idx += int(self.tree[node_idx, 3])
            
            # At a leaf node, return the prediction
            predictions[i] = self.tree[node_idx, 1]
            
        return predictions
        
    def get_feature_importances(self):
        """
        Calculate feature importance based on how frequently features are used for splitting.
        
        Returns:
        --------
        numpy.ndarray: Normalized feature importance scores
        """
        if not hasattr(self, 'tree') or self.tree is None:
            raise ValueError("Model has not been trained")
        
        if self.feature_count is None:
            raise ValueError("Feature count is not set")
        
        # Count feature usage in splits
        feature_counts = np.zeros(self.feature_count)
        
        # Iterate through the tree and count feature usage in splits
        # Tree structure: [feature, split_val, left_tree, right_tree]
        for i in range(len(self.tree)):
            node = self.tree[i]
            feature_idx = int(node[0])
            # If this is not a leaf node (indicated by feature >= 0)
            if feature_idx >= 0:
                feature_counts[feature_idx] += 1
        
        # Normalize to get importance
        total = np.sum(feature_counts)
        if total > 0:
            return feature_counts / total
        else:
            return np.zeros(self.feature_count)