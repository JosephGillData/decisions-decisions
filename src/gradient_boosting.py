"""
Gradient Boosting Implementation from Scratch

Gradient Boosting builds trees SEQUENTIALLY, where each tree corrects the
errors of the previous ones. This is fundamentally different from Random Forest
which builds trees INDEPENDENTLY.

Key Concepts
------------
1. SEQUENTIAL LEARNING: Each tree is trained on the RESIDUALS (errors) of the
   current model, not the original targets.

2. GRADIENT DESCENT IN FUNCTION SPACE: Instead of optimizing parameters, we're
   optimizing the function itself by adding new trees that follow the gradient.

3. LEARNING RATE (SHRINKAGE): New trees are scaled by a small factor (0.01-0.3)
   to prevent overfitting. Smaller rates need more trees but generalize better.

How It Works
------------
1. Start with a simple prediction: F_0(x) = mean(y)
2. For each iteration t:
   a. Compute residuals: r_t = y - F_{t-1}(x)
   b. Fit a tree h_t to predict these residuals
   c. Update model: F_t(x) = F_{t-1}(x) + learning_rate * h_t(x)
3. Final prediction: F_T(x) = F_0(x) + lr * sum(h_t(x))

Why It Works
------------
The residuals represent "what the current model got wrong". By training
each new tree to predict these errors, we're gradually correcting mistakes.
The learning rate prevents overcorrection.

Connection to Gradient Descent
------------------------------
For squared error loss L = (y - F(x))^2:
- The negative gradient is: -dL/dF = y - F(x) = residual
- So training on residuals IS following the negative gradient!
"""

import numpy as np
import pandas as pd
from typing import List, Union
from src.decision_tree import DecisionTree
from src.node import Node


class GradientBoosting:
  """
  Gradient Boosting regressor implemented from scratch.

  Builds trees sequentially, each one correcting errors of the ensemble.

  Parameters
  ----------
  n_estimators : int, default=100
    Number of boosting rounds (trees to add). More trees = more capacity
    but also more risk of overfitting.

  learning_rate : float, default=0.1
    Shrinkage factor applied to each tree. Lower values need more trees
    but typically generalize better. Common values: 0.01-0.3.

  criterion : str, default='squared_error'
    Split criterion for trees. 'squared_error' is standard for regression.

  max_depth : int, default=3
    Maximum depth of each tree. Gradient boosting typically uses SHALLOW
    trees (3-8 depth). Deeper trees can lead to overfitting.

  min_samples_split : int, default=10
    Minimum samples to split a node.

  min_samples_leaf : int, default=2
    Minimum samples in each leaf.

  verbose : bool, default=True
    Whether to print progress during training.

  Example
  -------
  >>> gb = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
  >>> model = gb.train(X_train, y_train)
  >>> predictions = gb.predict(model, X_test)
  """

  def __init__(
      self,
      n_estimators: int = 100,
      learning_rate: float = 0.1,
      criterion: str = 'squared_error',
      max_depth: int = 3,
      min_samples_split: int = 10,
      min_samples_leaf: int = 2,
      verbose: bool = True,
  ):
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
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

  def train(
      self,
      X_train: pd.DataFrame,
      y_train: pd.DataFrame
  ) -> List[Union[float, Node]]:
    """
    Train the gradient boosting model.

    The model is a list where:
    - model[0] = initial prediction (mean of targets)
    - model[1:] = decision tree root nodes

    Parameters
    ----------
    X_train : pd.DataFrame
      Feature matrix with shape (n_samples, n_features).
    y_train : pd.DataFrame
      Target values with shape (n_samples, 1).

    Returns
    -------
    list
      Model containing initial prediction and tree nodes.
    """
    target_col = y_train.columns[0]
    y_values = y_train[target_col].values

    # Step 1: Initialize with mean prediction
    # F_0(x) = mean(y) - simplest possible model
    initial_prediction = float(np.mean(y_values))
    current_predictions = np.full(len(y_values), initial_prediction)

    # Store the model: [initial_prediction, tree_1, tree_2, ...]
    model = [initial_prediction]

    # Step 2: Iteratively add trees
    for round_num in range(1, self.n_estimators + 1):
      if self.verbose:
        print(f"Boosting round {round_num}/{self.n_estimators}")

      # Compute residuals: what the current model gets wrong
      # For squared error, residuals = y - F(x) = negative gradient
      residuals = y_values - current_predictions
      residuals_df = pd.DataFrame({target_col: residuals})

      # Train a tree to predict the residuals
      tree = DecisionTree(
          criterion=self.criterion,
          max_depth=self.max_depth,
          min_samples_split=self.min_samples_split,
          min_samples_leaf=self.min_samples_leaf,
      )
      tree_root = tree.train(X_train, residuals_df)
      model.append(tree_root)

      # Update predictions: F_t = F_{t-1} + lr * h_t
      tree_predictions = tree.predict(tree_root, X_train)
      current_predictions += self.learning_rate * tree_predictions

    return model

  def predict(
      self,
      model: List[Union[float, Node]],
      X_test: pd.DataFrame
  ) -> np.ndarray:
    """
    Make predictions using the trained gradient boosting model.

    Final prediction = F_0 + lr * sum(h_t(x)) for all trees t

    Parameters
    ----------
    model : list
      Model returned by train().
    X_test : pd.DataFrame
      Feature matrix with shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
      Array of predictions with shape (n_samples,).
    """
    # Start with the initial prediction
    predictions = np.full(X_test.shape[0], model[0])

    # Add contribution from each tree
    for tree_root in model[1:]:
      tree_preds = self._tree_predictor.predict(tree_root, X_test)
      predictions += self.learning_rate * tree_preds

    return predictions
