import numpy as np
import pandas as pd
import logging
from python_scripts.node import Node


class DecisionTree:

    def __init__(
        self,
        problem_type: str = 'regression',
        criterion: str = 'variance',
        max_depth: int = 40,
        min_samples_split: int = 10,
        min_samples_leaf: int = 2,
    ):

        # Initialise the parameters
        self.problem_type = problem_type
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _gini_index(self, target: np.ndarray) -> float:
        """
        Calculate the gini index of an array.

        Args:
            target (np.ndarray): An array containing the classes of each sample.

        Returns:
            float: The gini index of the array.
        """

        # Get the unique values and their counts
        unique_values, counts = np.unique(target, return_counts=True)

        # Calculate the probabilities
        probabilities = counts / len(target)

        # Calculate the gini index
        gini = 1 - np.sum(probabilities**2)

        return gini

    def _variance(self, target: np.ndarray) -> float:
        """
        Calculate the variance of an array without using np.var.

        Args:
            target (np.ndarray): An array containing the value of each sample.

        Returns:
            float: The variance of the array.
        """

        prediction = np.mean(target)
        variance = sum((x - prediction) ** 2 for x in target) / len(target)

        return variance

    def _squared_error(self, target: np.ndarray) -> float:

        y_pred = np.mean(target)
        squared_error = sum((y_true - y_pred) ** 2 for y_true in target)

        return squared_error
    
    def _negative_squared_error_diff(self, target: np.ndarray) -> float:

        y_pred = np.mean(target)
        squared_error = sum((y_pred - y_true) for y_true in target)

        return squared_error

    def _split_node_criterion(self, left_node, right_node):

        left_node_count = len(left_node)
        right_node_count = len(right_node)
        parent_node_count = left_node_count + right_node_count

        # Calculate the gini indexes
        if self.criterion == "gini":
            left_node_criterion = self._gini_index(left_node)
            right_node_criterion = self._gini_index(right_node)

        elif self.criterion == "squared_error":
            left_node_criterion = self._squared_error(left_node)
            right_node_criterion = self._squared_error(right_node)

        elif self.criterion == "negative_squared_error_diff":
            left_node_criterion = self._negative_squared_error_diff(left_node)
            right_node_criterion = self._negative_squared_error_diff(right_node)

        elif self.criterion == "variance":
            left_node_criterion = self._variance(left_node)
            right_node_criterion = self._variance(right_node)

        # fmt: off
        weighted_split_node_criterion = (left_node_count / parent_node_count) * left_node_criterion \
            + (right_node_count / parent_node_count) * right_node_criterion
        # fmt: on

        return weighted_split_node_criterion

    def _optimal_split(self, values_targets):
        """
        Iterate over the values of the column to identify the optimal split for that column.

        Args:
            values_targets (np.array): Contains the values of the split column and the target values.

        Returns:
            float: The value in the column that has the optimal split
            float: The criterion in the split_node of the optimal split
        """

        values_targets = values_targets[values_targets[:, 0].argsort()]
        unique_values = np.sort(np.unique(values_targets[:, 0]))

        min_split_criterion = np.inf
        optimal_value = None
        left_node = np.array([])
        right_node = unique_values

        # Iterate over all possible splits of the unique values
        # Ensure both left_node and right_node are non-empty
        for value in unique_values[:-1]:

            # Apply the split to create two child nodes
            left_node = values_targets[values_targets[:, 0] <= value][:, 1]
            right_node = values_targets[values_targets[:, 0] > value][:, 1]

            # Calculate the criterion value of the split
            split_criterion = self._split_node_criterion(left_node, right_node)

            # Update optimal column split,
            if split_criterion < min_split_criterion:
                min_split_criterion = split_criterion
                optimal_value = value

        return optimal_value, min_split_criterion

    def _identify_optimal_split(self, data):
        """
        Takes a node (X and y), iterates over the columns of X to identify the optimal split, such that it
        minimises variance of the target values in the split nodes.

        Args:
            X (np.array): The X values of the node
            y (np.array): _description_

        Returns:
            _type_: _description_
        """

        global_optimum_column_idx = None
        global_optimum_value = None
        global_min_split_criterion = np.inf

        X = data[:, :-1]  # All columns except the last
        y = data[:, -1]  # Only the last column

        # Iterate over all columns in X
        for column_idx in range(X.shape[1]):
            # Values of the column
            column_values = X[:, column_idx]
            # Combine the split column values with the target column values
            column_values_targets = np.array([column_values, y]).T
            # Calculate the optimal split for the column
            column_optimal_value, column_min_split_criterion = self._optimal_split(column_values_targets)

            # TODO write a verbose logic to capture data at each split

            # Store the best min_split_criterion over all columns
            if column_min_split_criterion < global_min_split_criterion:
                global_optimum_column_idx = column_idx
                global_optimum_value = column_optimal_value
                global_min_split_criterion = column_min_split_criterion

        return global_optimum_column_idx, global_optimum_value

    def _apply_optimal_split(self, data, global_optimum_column_idx, global_optimum_value):

        # Create left and right nodes for rows above and below the optimal split value
        left_data = data[data[:, global_optimum_column_idx] <= global_optimum_value]
        right_data = data[data[:, global_optimum_column_idx] > global_optimum_value]

        # Return nodes
        return left_data, right_data

    def _create_leaf_node_value(self, data):

        y = data[:, -1]

        return np.mean(y)

    def _create_decision_node(self, data, tree_depth=0, side=None):

        self.node_count += 1

        # Create child nodes if conditions met
        if (tree_depth <= self.max_depth) & (data.shape[0] >= self.min_samples_split):

            # Identify and apply optimal split
            global_optimum_column_idx, global_optimum_value = self._identify_optimal_split(data)
            left_data, right_data = self._apply_optimal_split(data, global_optimum_column_idx, global_optimum_value)

            # Initialise the node with the data and split information
            node = Node(data=data, feature_idx=global_optimum_column_idx, threshold=global_optimum_value)

            # If either of the children are too small, return the node with no child data (leaf nodes)
            # node.left_child_node and node.right_child_node are None
            if (left_data.shape[0] <= self.min_samples_leaf) | (right_data.shape[0] <= self.min_samples_leaf):
                node.leaf_node_value = self._create_leaf_node_value(data)
                node.is_leaf = True
                return node

            # If the children are large enough, turn them into decision nodes
            tree_depth += 1
            node.left_child_node = self._create_decision_node(left_data, tree_depth, 'left')
            node.right_child_node = self._create_decision_node(right_data, tree_depth, 'right')

        # Node is a leaf node
        else:
            leaf_node_value = self._create_leaf_node_value(data)
            node = Node(leaf_node_value=leaf_node_value)
            node.is_leaf = True

        return node

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):

        self.node_count = 0

        # Initialise global variables
        self.target_value = y_train.columns[0]

        # Join the X and y data
        data = np.concatenate((X_train, np.reshape(y_train, (-1, 1))), axis=1)

        # Create the decision tree
        # Function calls itself, iteratively building out the tree
        decision_tree_model = self._create_decision_node(data)

        return decision_tree_model
    
    def _single_prediction(self, x, model):

        if model.is_leaf:
            return model.leaf_node_value
        
        column_name = x.keys()[model.feature_idx]

        if (self.X_dtypes.values[model.feature_idx] == int) | (self.X_dtypes.values[model.feature_idx] == float):
            if x[column_name] <= model.threshold:
                return self._single_prediction(x, model.left_child_node)
            else:
                return self._single_prediction(x, model.right_child_node)
            
        if self.X_dtypes.values[model.feature_idx] == object:
            if x[column_name] == model.threshold:
                return self._single_prediction(x, model.left_child_node)
            else:
                return self._single_prediction(x, model.right_child_node)

    def predict(self, model, X_test: pd.DataFrame):

        self.X_dtypes = X_test.dtypes
        predictions = [float(self._single_prediction(x, model)) for i, x in X_test.iterrows()]
        predictions = np.array(predictions)

        return predictions

        
    
