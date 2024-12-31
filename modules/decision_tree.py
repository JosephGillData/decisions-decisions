import numpy as np
import pandas as pd
import logging


class DecisionTree:

    def __init__(
        self,
        criterion: str,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
    ):

        # Initialise the parameters
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

    def _gini_index_split(self, left_node, right_node):

        left_node_count = len(left_node)
        right_node_count = len(right_node)
        parent_node_count = left_node_count + right_node_count

        # Calculate the gini indexes
        left_node_gini = self._gini_index(left_node)
        right_node_gini = self._gini_index(right_node)

        # fmt: off
        weighted_gini_index = (left_node_count / parent_node_count) * left_node_gini \
            + (right_node_count / parent_node_count) * right_node_gini
        # fmt: on

        return weighted_gini_index

    def _optimal_split(self, values_targets):

        values_targets = values_targets[values_targets[:, 0].argsort()]
        unique_values = np.sort(np.unique(values_targets[:, 0]))

        min_gini_split = 1
        optimal_value = None
        left_node = np.array([])
        right_node = unique_values

        # Iterate over all possible splits of the unique values
        # Ensure both left_node and right_node are non-empty
        for value in unique_values[:-1]:

            # Apply the split to create two child nodes
            left_node = values_targets[values_targets[:, 0] <= value][:, 1]
            right_node = values_targets[values_targets[:, 0] > value][:, 1]

            # Calculate the Gini index of the split
            gini_split = self._gini_index_split(left_node, right_node)

            if gini_split < min_gini_split:
                min_gini_split = gini_split
                optimal_value = value

        return optimal_value, min_gini_split

    # TODO write this function
    def _apply_split(self, X, y):

        global_optimum_column_index = None
        global_optimum_value = None
        global_min_gini_split = 1

        for column_index in range(X.shape[1]):  # range(1):  #
            column_values = X[:, column_index]
            column_values_targets = np.array([column_values, y]).T
            column_optimal_value, column_min_gini_split = self._optimal_split(
                column_values_targets
            )

            if column_min_gini_split < global_min_gini_split:
                global_optimum_column_index = column_index
                global_optimum_value = column_optimal_value
                global_min_gini_split = column_min_gini_split

        global_values_targets = np.array([X[:, global_optimum_column_index], y]).T

        left_X = X[X[:, global_optimum_column_index] <= global_optimum_value]
        right_X = X[X[:, global_optimum_column_index] > global_optimum_value]

        left_y = global_values_targets[
            global_values_targets[:, 0] <= global_optimum_value
        ]
        right_y = global_values_targets[
            global_values_targets[:, 0] > global_optimum_value
        ]

        return left_X, right_X, left_y, right_y

    def train(self, X_df: pd.DataFrame, y_df: pd.DataFrame):

        tree_depth = 0

        X_columns = X_df.columns
        X = X_df.values
        y_values = y_df.values

        # TODO need to track the tree basically, but first apply a single split

        while tree_depth < self.max_depth:
            tree_depth = tree_depth + 1
            print("Tree Depth", tree_depth)

            left_X, right_X, left_y, right_y = self._apply_split(X, y_values)

            print(np.shape(left_X))
            print(np.shape(right_X))
            print(np.shape(left_y))
            print(np.shape(right_y))
