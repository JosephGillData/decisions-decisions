"""
Random Forest Implementation from Scratch

Random Forest is an ensemble method that builds multiple decision trees and
averages their predictions. This reduces overfitting compared to a single tree.

Key Concepts
------------
1. BAGGING (Bootstrap Aggregating): Each tree is trained on a random sample
   of the data WITH replacement. This creates diversity between trees.

2. ENSEMBLE AVERAGING: Final prediction is the mean of all tree predictions.
   This reduces variance and makes the model more robust.

3. WISDOM OF THE CROWD: Many weak learners (trees) together can outperform
   a single strong learner, especially when they make uncorrelated errors.

Why Random Forest Works
-----------------------
- Individual trees might overfit to noise in their training sample
- But noise is random - different trees overfit to different noise
- When we average predictions, the noise cancels out, leaving the signal

Mathematical Foundation
-----------------------
For regression, the prediction is simply:
  y_hat = (1/T) * sum(tree_t(x)) for t = 1 to T

Where T is the number of trees and tree_t(x) is the prediction of tree t.
"""

import numpy as np
import pandas as pd
from typing import List
from src.decision_tree import DecisionTree
from src.node import Node


class RandomForest:
  """
  Random Forest ensemble for regression (easily extendable to classification).

  Builds multiple decision trees on bootstrap samples and averages predictions.

  Parameters
  ----------
  n_estimators : int, default=100
    Number of trees in the forest. More trees generally improve accuracy
    but increase training time. After ~100-200 trees, returns diminish.

  criterion : str, default='variance'
    Split criterion for individual trees. See DecisionTree for options.

  max_depth : int, default=10
    Maximum depth of each tree. Shallower trees = more regularization.

  min_samples_split : int, default=5
    Minimum samples needed to split a node.

  min_samples_leaf : int, default=2
    Minimum samples required in each leaf.

  verbose : bool, default=True
    Whether to print progress during training.

  Example
  -------
  >>> rf = RandomForest(n_estimators=50, max_depth=5)
  >>> model = rf.train(X_train, y_train)
  >>> predictions = rf.predict(model, X_test)
  """

  def __init__(
      self,
      n_estimators: int = 100,
      criterion: str = 'variance',
      max_depth: int = 10,
      min_samples_split: int = 5,
      min_samples_leaf: int = 2,
      verbose: bool = True,
  ):
    self.n_estimators = n_estimators
    self.criterion = criterion
    self.max_depth = max_depth
    self.min_samples_split = min_samples_split
    self.min_samples_leaf = min_samples_leaf
    self.verbose = verbose

    # Store a DecisionTree instance for predictions
    self._tree_predictor = DecisionTree(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
    )

  def _create_bootstrap_sample(
      self,
      X: pd.DataFrame,
      y: pd.DataFrame
  ) -> tuple:
    """
    Create a bootstrap sample (sampling with replacement).

    Bootstrap sampling is key to bagging. Each tree sees a slightly
    different dataset, which creates diversity in the ensemble.

    On average, each bootstrap sample contains ~63.2% unique samples
    (the rest are duplicates). Samples not included can be used for
    out-of-bag error estimation.

    Parameters
    ----------
    X : pd.DataFrame
      Feature matrix.
    y : pd.DataFrame
      Target values.

    Returns
    -------
    tuple
      (X_sample, y_sample): Bootstrap sample of features and targets.
    """
    combined = pd.concat([X, y], axis=1)
    target_col = y.columns[0]

    # Sample with replacement (same size as original)
    sample = combined.sample(frac=1, replace=True)

    X_sample = sample.drop(columns=[target_col])
    y_sample = pd.DataFrame(sample[target_col])

    return X_sample, y_sample

  def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> List[Node]:
    """
    Train the random forest by building multiple decision trees.

    Each tree is trained on a bootstrap sample of the data.

    Parameters
    ----------
    X_train : pd.DataFrame
      Feature matrix with shape (n_samples, n_features).
    y_train : pd.DataFrame
      Target values with shape (n_samples, 1).

    Returns
    -------
    list of Node
      List of trained tree root nodes.
    """
    trees = []

    for tree_idx in range(1, self.n_estimators + 1):
      if self.verbose:
        print(f"Training tree {tree_idx}/{self.n_estimators}")

      # Create bootstrap sample for this tree
      X_sample, y_sample = self._create_bootstrap_sample(X_train, y_train)

      # Train a decision tree on the bootstrap sample
      tree = DecisionTree(
          criterion=self.criterion,
          max_depth=self.max_depth,
          min_samples_split=self.min_samples_split,
          min_samples_leaf=self.min_samples_leaf,
      )
      root = tree.train(X_sample, y_sample)
      trees.append(root)

    return trees

  def predict(self, model: List[Node], X_test: pd.DataFrame) -> np.ndarray:
    """
    Make predictions by averaging predictions from all trees.

    Parameters
    ----------
    model : list of Node
      List of trained tree root nodes (returned by train()).
    X_test : pd.DataFrame
      Feature matrix with shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
      Array of predictions with shape (n_samples,).
      Each prediction is the mean of all tree predictions.
    """
    # Collect predictions from each tree
    all_predictions = []

    for tree_root in model:
      tree_preds = self._tree_predictor.predict(tree_root, X_test)
      all_predictions.append(tree_preds)

    # Average predictions across all trees
    all_predictions = np.array(all_predictions)
    ensemble_predictions = all_predictions.mean(axis=0)

    return ensemble_predictions
