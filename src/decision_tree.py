"""
Decision Tree Implementation from Scratch

This module implements a decision tree algorithm for both classification and regression
problems without using sklearn. The goal is to demonstrate understanding of the core
concepts: recursive partitioning, impurity measures, and greedy split selection.

Key Concepts
------------
1. SPLITTING: At each node, we find the feature and threshold that best separates
   the data. "Best" means minimizing impurity (classification) or variance (regression).

2. IMPURITY MEASURES: We use Gini index for classification and variance for regression.
   Lower impurity = more homogeneous groups = better predictions.

3. RECURSION: The tree builds itself by recursively creating child nodes until
   stopping conditions are met (max depth, min samples, pure nodes).

4. GREEDY ALGORITHM: We find the locally optimal split at each node. This doesn't
   guarantee a globally optimal tree, but is computationally tractable.

Author: Joseph
Purpose: Educational / Portfolio project demonstrating ML fundamentals
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from src.node import Node


class DecisionTree:
  """
  A decision tree implementation for classification and regression.

  The tree works by recursively partitioning data based on feature thresholds.
  At each node, we select the split that minimizes the chosen impurity criterion.

  Parameters
  ----------
  criterion : str, default='variance'
    The function to measure split quality.
    - 'gini': Gini impurity for classification. Measures probability of
      misclassifying a randomly chosen element.
    - 'variance': Variance reduction for regression. Measures spread of
      target values around the mean.
    - 'squared_error': Sum of squared errors for regression (used in
      gradient boosting where we want absolute error, not normalized).

  max_depth : int, default=10
    Maximum depth of the tree. Deeper trees can capture more complex patterns
    but risk overfitting. This is the primary regularization parameter.

  min_samples_split : int, default=10
    Minimum samples required to split a node. If a node has fewer samples,
    it becomes a leaf. Prevents creating splits on tiny subsets.

  min_samples_leaf : int, default=2
    Minimum samples required in each leaf after a split. If a proposed split
    would create a leaf with fewer samples, the split is rejected.

  Attributes
  ----------
  node_count : int
    Total number of nodes in the tree (set after calling train()).

  Example
  -------
  >>> tree = DecisionTree(criterion='variance', max_depth=5)
  >>> model = tree.train(X_train, y_train)
  >>> predictions = tree.predict(model, X_test)
  """

  def __init__(
      self,
      criterion: str = 'variance',
      max_depth: int = 10,
      min_samples_split: int = 10,
      min_samples_leaf: int = 2,
  ):
    self.criterion = criterion
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.min_samples_leaf = min_samples_leaf

    # Set during training
    self.node_count: int = 0
    self._feature_dtypes: Optional[pd.Series] = None
    self._feature_names: Optional[list] = None

  # ===========================================================================
  # IMPURITY MEASURES
  # ===========================================================================
  # These functions quantify how "mixed" or "spread out" the target values are
  # in a node. Lower values = more homogeneous = better for making predictions.
  # ===========================================================================

  def _gini_impurity(self, targets: np.ndarray) -> float:
    """
    Calculate Gini impurity for classification problems.

    Gini impurity measures the probability of incorrectly classifying a
    randomly chosen element if it were labeled according to the distribution
    of labels in the node.

    Formula: Gini = 1 - sum(p_i^2) for each class i

    Why this formula?
    - p_i^2 is the probability of picking two samples of class i
    - sum(p_i^2) is the probability of picking two samples of the same class
    - 1 - sum(p_i^2) is the probability of picking two different classes

    Interpretation:
    - If all samples belong to one class: Gini = 0 (pure node, perfect)
    - If samples are evenly split across classes: Gini approaches 0.5 (for 2 classes)

    Parameters
    ----------
    targets : np.ndarray
      Array of class labels for samples in this node.

    Returns
    -------
    float
      Gini impurity value between 0 (pure) and 0.5 (maximally mixed for 2 classes).
    """
    if len(targets) == 0:
      return 0.0

    _, counts = np.unique(targets, return_counts=True)
    probabilities = counts / len(targets)

    gini = 1 - np.sum(probabilities ** 2)
    return gini

  def _variance(self, targets: np.ndarray) -> float:
    """
    Calculate variance for regression problems.

    Variance measures how spread out the target values are around the mean.
    In a decision tree context, we want nodes where the target values are
    similar (low variance), so predictions (the mean) are accurate.

    Formula: Var = (1/n) * sum((y_i - mean)^2)

    Why variance for regression?
    - The decision tree predicts the mean of targets in a leaf node
    - Variance directly measures how far samples are from that prediction
    - Minimizing variance = minimizing prediction error

    Parameters
    ----------
    targets : np.ndarray
      Array of continuous target values for samples in this node.

    Returns
    -------
    float
      Variance of target values. Lower = more homogeneous predictions.
    """
    if len(targets) == 0:
      return 0.0

    mean = np.mean(targets)
    variance = np.sum((targets - mean) ** 2) / len(targets)
    return variance

  def _squared_error(self, targets: np.ndarray) -> float:
    """
    Calculate sum of squared errors for regression.

    Similar to variance but returns the total squared error, not normalized.
    Used in gradient boosting where we care about absolute error reduction,
    not the average error per sample.

    Parameters
    ----------
    targets : np.ndarray
      Array of continuous target values.

    Returns
    -------
    float
      Sum of squared errors from the mean.
    """
    if len(targets) == 0:
      return 0.0

    mean = np.mean(targets)
    return np.sum((targets - mean) ** 2)

  def _compute_impurity(self, targets: np.ndarray) -> float:
    """
    Compute impurity using the configured criterion.

    This is the dispatch function that routes to the appropriate
    impurity measure based on self.criterion.

    Parameters
    ----------
    targets : np.ndarray
      Target values for samples in a node.

    Returns
    -------
    float
      Impurity value (lower is better).

    Raises
    ------
    ValueError
      If criterion is not recognized.
    """
    if self.criterion == 'gini':
      return self._gini_impurity(targets)
    elif self.criterion == 'variance':
      return self._variance(targets)
    elif self.criterion == 'squared_error':
      return self._squared_error(targets)
    else:
      raise ValueError(f"Unknown criterion: {self.criterion}")

  # ===========================================================================
  # SPLIT EVALUATION
  # ===========================================================================
  # These functions evaluate how good a potential split is by measuring the
  # weighted impurity of the resulting child nodes.
  # ===========================================================================

  def _weighted_impurity_of_split(
      self,
      left_targets: np.ndarray,
      right_targets: np.ndarray
  ) -> float:
    """
    Calculate the weighted impurity after splitting into left and right nodes.

    When evaluating a split, we compute impurity for each child node and
    weight them by the proportion of samples they contain. This ensures
    we prefer splits that create large, pure nodes over tiny pure ones.

    Formula: weighted_impurity = (n_left/n_total) * impurity_left
                               + (n_right/n_total) * impurity_right

    Why weighted average?
    - A split creating one pure node with 1 sample and one impure node with 99
      samples is worse than a split creating two moderately pure nodes with 50 each
    - Weighting by sample count reflects this preference

    Parameters
    ----------
    left_targets : np.ndarray
      Target values for samples that would go to the left child.
    right_targets : np.ndarray
      Target values for samples that would go to the right child.

    Returns
    -------
    float
      Weighted average impurity of the split. Lower = better split.
    """
    n_left = len(left_targets)
    n_right = len(right_targets)
    n_total = n_left + n_right

    if n_total == 0:
      return np.inf

    left_impurity = self._compute_impurity(left_targets)
    right_impurity = self._compute_impurity(right_targets)

    weighted_impurity = (
        (n_left / n_total) * left_impurity +
        (n_right / n_total) * right_impurity
    )
    return weighted_impurity

  def _find_best_threshold_for_feature(
      self,
      feature_values: np.ndarray,
      targets: np.ndarray
  ) -> Tuple[Optional[float], float]:
    """
    Find the best split threshold for a single feature.

    We try every unique value as a potential threshold and select the one
    that minimizes weighted impurity. This is the brute-force approach that
    guarantees finding the optimal threshold for this feature.

    Threshold semantics: samples with feature_value <= threshold go left,
    samples with feature_value > threshold go right.

    Why try every unique value?
    - For continuous features, the optimal threshold is always at a data point
    - Trying midpoints between values would be equivalent but more complex
    - This guarantees we find the globally optimal threshold for this feature

    Parameters
    ----------
    feature_values : np.ndarray
      Values of a single feature for all samples in the node.
    targets : np.ndarray
      Target values for all samples in the node.

    Returns
    -------
    tuple
      (best_threshold, best_impurity): The threshold value that minimizes
      impurity and the resulting weighted impurity. Returns (None, inf) if
      no valid split exists.
    """
    # Sort by feature value to efficiently iterate through thresholds
    sorted_indices = feature_values.argsort()
    sorted_features = feature_values[sorted_indices]
    sorted_targets = targets[sorted_indices]

    unique_values = np.unique(sorted_features)

    # Need at least 2 unique values to make a split
    if len(unique_values) < 2:
      return None, np.inf

    best_threshold = None
    best_impurity = np.inf

    # Try each unique value as a threshold (except the last one)
    # We skip the last because threshold >= max would put nothing in right node
    for threshold in unique_values[:-1]:
      # Split: left gets values <= threshold, right gets values > threshold
      left_mask = sorted_features <= threshold
      left_targets = sorted_targets[left_mask]
      right_targets = sorted_targets[~left_mask]

      # Compute weighted impurity of this split
      impurity = self._weighted_impurity_of_split(left_targets, right_targets)

      if impurity < best_impurity:
        best_impurity = impurity
        best_threshold = threshold

    return best_threshold, best_impurity

  def _find_best_split(
      self,
      data: np.ndarray
  ) -> Tuple[Optional[int], Optional[float]]:
    """
    Find the best feature and threshold to split the node.

    This is the core of the GREEDY algorithm: we evaluate all features,
    find the best threshold for each, and select the overall best split.

    Why greedy?
    - Finding the globally optimal tree is NP-hard
    - Greedy approach (best local choice at each node) is tractable
    - Works well in practice despite not guaranteeing global optimum

    Parameters
    ----------
    data : np.ndarray
      Combined feature matrix and target column. Shape: (n_samples, n_features + 1)
      The last column contains target values.

    Returns
    -------
    tuple
      (best_feature_idx, best_threshold): Index of the feature to split on
      and the threshold value. Returns (None, None) if no valid split exists.
    """
    features = data[:, :-1]  # All columns except last
    targets = data[:, -1]    # Last column is target

    best_feature_idx = None
    best_threshold = None
    best_impurity = np.inf

    # Evaluate each feature to find the one with the best split
    for feature_idx in range(features.shape[1]):
      feature_values = features[:, feature_idx]
      threshold, impurity = self._find_best_threshold_for_feature(
          feature_values, targets
      )

      if impurity < best_impurity:
        best_impurity = impurity
        best_feature_idx = feature_idx
        best_threshold = threshold

    return best_feature_idx, best_threshold

  def _apply_split(
      self,
      data: np.ndarray,
      feature_idx: int,
      threshold: float
  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into left and right subsets based on feature and threshold.

    Convention: left child gets samples where feature <= threshold,
    right child gets samples where feature > threshold.

    Parameters
    ----------
    data : np.ndarray
      Combined feature matrix and target column.
    feature_idx : int
      Index of the feature to split on.
    threshold : float
      Threshold value for the split.

    Returns
    -------
    tuple
      (left_data, right_data): Data subsets for left and right child nodes.
    """
    left_mask = data[:, feature_idx] <= threshold
    left_data = data[left_mask]
    right_data = data[~left_mask]
    return left_data, right_data

  # ===========================================================================
  # TREE CONSTRUCTION (RECURSIVE)
  # ===========================================================================
  # The tree is built recursively. Each call to _build_node creates one node
  # and potentially recurses to create child nodes.
  # ===========================================================================

  def _compute_leaf_value(self, targets: np.ndarray) -> float:
    """
    Compute the prediction value for a leaf node.

    For both regression and classification (with numeric labels), we return
    the mean of target values. For classification, this works when classes
    are encoded as 0/1 - the mean represents the probability of class 1.

    Parameters
    ----------
    targets : np.ndarray
      Target values for samples in this leaf.

    Returns
    -------
    float
      The prediction value for this leaf.
    """
    return np.mean(targets)

  def _build_node(self, data: np.ndarray, depth: int = 0) -> Node:
    """
    Recursively build the decision tree.

    This is the heart of the algorithm. For each node, we:
    1. Check stopping conditions (max depth, min samples)
    2. Find the best split
    3. Check if the split creates valid child nodes
    4. Recurse on child nodes OR create a leaf

    STOPPING CONDITIONS (when we create a leaf instead of splitting):
    - Reached max_depth: Prevents overfitting by limiting tree complexity
    - Too few samples (< min_samples_split): Not enough data for meaningful split
    - Split would create tiny leaves (< min_samples_leaf): Regularization
    - No valid split exists: All features have single unique value

    Parameters
    ----------
    data : np.ndarray
      Combined features and target. Shape: (n_samples, n_features + 1)
    depth : int
      Current depth in the tree (0 = root).

    Returns
    -------
    Node
      The constructed node (may be internal node or leaf).
    """
    self.node_count += 1
    n_samples = data.shape[0]
    targets = data[:, -1]

    # -----------------------------------------------------------------
    # STOPPING CONDITION 1: Depth or sample count limits reached
    # -----------------------------------------------------------------
    if depth >= self.max_depth or n_samples < self.min_samples_split:
      return Node(
          leaf_node_value=self._compute_leaf_value(targets),
          is_leaf=True
      )

    # -----------------------------------------------------------------
    # FIND THE BEST SPLIT (greedy selection)
    # -----------------------------------------------------------------
    best_feature_idx, best_threshold = self._find_best_split(data)

    # -----------------------------------------------------------------
    # STOPPING CONDITION 2: No valid split found
    # -----------------------------------------------------------------
    if best_feature_idx is None:
      return Node(
          leaf_node_value=self._compute_leaf_value(targets),
          is_leaf=True
      )

    # -----------------------------------------------------------------
    # APPLY THE SPLIT to create child data subsets
    # -----------------------------------------------------------------
    left_data, right_data = self._apply_split(
        data, best_feature_idx, best_threshold
    )

    # -----------------------------------------------------------------
    # STOPPING CONDITION 3: Split would create too-small leaves
    # -----------------------------------------------------------------
    if (len(left_data) < self.min_samples_leaf or
        len(right_data) < self.min_samples_leaf):
      return Node(
          leaf_node_value=self._compute_leaf_value(targets),
          is_leaf=True
      )

    # -----------------------------------------------------------------
    # RECURSE: Build child nodes
    # -----------------------------------------------------------------
    node = Node(
        feature_idx=best_feature_idx,
        threshold=best_threshold,
        n_samples=n_samples
    )
    node.left_child_node = self._build_node(left_data, depth + 1)
    node.right_child_node = self._build_node(right_data, depth + 1)

    return node

  # ===========================================================================
  # PUBLIC API: train() and predict()
  # ===========================================================================

  def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Node:
    """
    Train the decision tree on the provided data.

    The training process builds the tree recursively by finding optimal
    splits at each node until stopping conditions are met.

    Parameters
    ----------
    X_train : pd.DataFrame
      Feature matrix with shape (n_samples, n_features).
    y_train : pd.DataFrame
      Target values with shape (n_samples, 1).

    Returns
    -------
    Node
      The root node of the trained tree.
    """
    self.node_count = 0
    self._feature_dtypes = X_train.dtypes
    self._feature_names = list(X_train.columns)

    # Combine X and y into single array for efficient slicing during recursion
    # This avoids repeatedly aligning separate X and y arrays
    data = np.column_stack([X_train.values, y_train.values.ravel()])

    root = self._build_node(data)
    return root

  def _predict_single(self, sample: pd.Series, node: Node) -> float:
    """
    Traverse the tree to make a prediction for a single sample.

    Starting from the root, we follow left/right branches based on the
    sample's feature values until we reach a leaf node.

    Parameters
    ----------
    sample : pd.Series
      A single sample with feature values.
    node : Node
      Current node in traversal (starts at root).

    Returns
    -------
    float
      Predicted value from the leaf node.
    """
    # Base case: reached a leaf node
    if node.is_leaf:
      return node.leaf_node_value

    # Get the feature value and type for the split feature

    feature_name = self._feature_names[node.feature_idx]
    feature_value = sample[feature_name]
    feature_dtype = self._feature_dtypes[feature_name]

    # Navigate to appropriate child based on feature type
    if feature_dtype in [int, float, np.int64, np.float64]:
      # Numeric feature: compare with threshold
      if feature_value <= node.threshold:
        return self._predict_single(sample, node.left_child_node)
      else:
        return self._predict_single(sample, node.right_child_node)
    else:
      # Categorical feature: equality check (left if match, right otherwise)
      if feature_value == node.threshold:
        return self._predict_single(sample, node.left_child_node)
      else:
        return self._predict_single(sample, node.right_child_node)

  def predict(self, model: Node, X_test: pd.DataFrame) -> np.ndarray:
    """
    Make predictions for multiple samples.

    Parameters
    ----------
    model : Node
      The root node of a trained tree (returned by train()).
    X_test : pd.DataFrame
      Feature matrix with shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
      Array of predictions with shape (n_samples,).
    """

    self._feature_names = list(X_test.columns)

    self._feature_dtypes = X_test.dtypes
    for _, row in X_test.iterrows():
      predictions = [float(self._predict_single(row, model))]
    return np.array(predictions)
