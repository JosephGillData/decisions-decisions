"""
Unit tests for Decision Tree implementation.

These tests verify the core logic of the decision tree algorithm:
- Impurity calculations (Gini, variance)
- Split selection
- Tree building and prediction
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.decision_tree import DecisionTree
from src.node import Node


class TestImpurityMeasures:
  """Test the impurity calculation functions."""

  def test_gini_pure_node(self):
    """Gini impurity of a pure node (all same class) should be 0."""
    tree = DecisionTree(criterion='gini')
    targets = np.array([1, 1, 1, 1, 1])
    assert tree._gini_impurity(targets) == 0.0

  def test_gini_mixed_node(self):
    """Gini impurity of 50/50 split should be 0.5."""
    tree = DecisionTree(criterion='gini')
    targets = np.array([0, 0, 1, 1])
    assert tree._gini_impurity(targets) == 0.5

  def test_gini_unbalanced(self):
    """Gini impurity of 75/25 split should be 0.375."""
    tree = DecisionTree(criterion='gini')
    targets = np.array([0, 0, 0, 1])
    expected = 1 - (0.75**2 + 0.25**2)  # 1 - 0.5625 - 0.0625 = 0.375
    assert abs(tree._gini_impurity(targets) - expected) < 1e-10

  def test_variance_constant(self):
    """Variance of constant values should be 0."""
    tree = DecisionTree(criterion='variance')
    targets = np.array([5.0, 5.0, 5.0, 5.0])
    assert tree._variance(targets) == 0.0

  def test_variance_simple(self):
    """Test variance calculation with known values."""
    tree = DecisionTree(criterion='variance')
    targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Mean = 3, variance = mean of squared differences
    # = (4 + 1 + 0 + 1 + 4) / 5 = 2
    assert tree._variance(targets) == 2.0

  def test_empty_array(self):
    """Empty arrays should return 0 impurity."""
    tree = DecisionTree(criterion='variance')
    assert tree._variance(np.array([])) == 0.0
    tree = DecisionTree(criterion='gini')
    assert tree._gini_impurity(np.array([])) == 0.0


class TestSplitSelection:
  """Test the split finding logic."""

  def test_weighted_impurity(self):
    """Test weighted impurity calculation."""
    tree = DecisionTree(criterion='variance')
    left = np.array([1.0, 1.0])   # variance = 0
    right = np.array([5.0, 5.0])  # variance = 0

    weighted = tree._weighted_impurity_of_split(left, right)
    assert weighted == 0.0

  def test_find_best_threshold(self):
    """Test finding optimal threshold for a feature."""
    tree = DecisionTree(criterion='variance')

    # Feature values that clearly split the targets
    feature_values = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    targets = np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0])

    threshold, impurity = tree._find_best_threshold_for_feature(
        feature_values, targets
    )

    # Best split should be between 3 and 10
    assert threshold == 3.0
    # After split, both sides have variance 0
    assert impurity == 0.0

  def test_no_valid_split(self):
    """When all values are the same, no split is possible."""
    tree = DecisionTree(criterion='variance')

    feature_values = np.array([5.0, 5.0, 5.0, 5.0])
    targets = np.array([1.0, 2.0, 3.0, 4.0])

    threshold, impurity = tree._find_best_threshold_for_feature(
        feature_values, targets
    )

    assert threshold is None
    assert impurity == np.inf


class TestTreeBuilding:
  """Test the full tree building process."""

  def test_train_simple_regression(self):
    """Train a tree on simple regression data."""
    tree = DecisionTree(
        criterion='variance',
        max_depth=2,
        min_samples_split=2,
        min_samples_leaf=1
    )

    # Simple dataset: high values when feature > 5
    X = pd.DataFrame({
        'feature': [1, 2, 3, 4, 8, 9, 10, 11]
    })
    y = pd.DataFrame({
        'target': [10, 11, 12, 13, 90, 91, 92, 93]
    })

    model = tree.train(X, y)

    assert model is not None
    assert isinstance(model, Node)
    assert tree.node_count > 0

  def test_predict_returns_array(self):
    """Predictions should return numpy array."""
    tree = DecisionTree(max_depth=2, min_samples_split=2, min_samples_leaf=1)

    X_train = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
    y_train = pd.DataFrame({'y': [10, 20, 30, 40, 50]})
    X_test = pd.DataFrame({'x': [2.5, 4.5]})

    model = tree.train(X_train, y_train)
    predictions = tree.predict(model, X_test)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 2

  def test_max_depth_respected(self):
    """Tree should not exceed max_depth."""
    tree = DecisionTree(
        max_depth=1,  # Very shallow
        min_samples_split=2,
        min_samples_leaf=1
    )

    X = pd.DataFrame({'x': range(100)})
    y = pd.DataFrame({'y': range(100)})

    model = tree.train(X, y)

    # With max_depth=1, we should have at most 3 nodes (root + 2 leaves)
    assert tree.node_count <= 3

  def test_leaf_node_prediction(self):
    """Test that leaf nodes return their stored value."""
    leaf = Node(leaf_node_value=42.0, is_leaf=True)
    assert leaf.is_leaf
    assert leaf.leaf_node_value == 42.0


class TestPrediction:
  """Test prediction accuracy on known data."""

  def test_perfect_split(self):
    """Tree should perfectly split clearly separable data."""
    tree = DecisionTree(
        criterion='variance',
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1
    )

    # Data where feature <= 5 has target ~10, feature > 5 has target ~100
    X_train = pd.DataFrame({
        'feature': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    })
    y_train = pd.DataFrame({
        'target': [10, 10, 10, 10, 10, 100, 100, 100, 100, 100]
    })

    model = tree.train(X_train, y_train)

    # Test predictions
    X_test = pd.DataFrame({'feature': [3, 8]})
    predictions = tree.predict(model, X_test)

    # Prediction for feature=3 should be ~10
    assert predictions[0] == 10.0
    # Prediction for feature=8 should be ~100
    assert predictions[1] == 100.0


class TestNode:
  """Test the Node class."""

  def test_internal_node_repr(self):
    """Internal node should show feature and threshold."""
    node = Node(feature_idx=0, threshold=5.5)
    assert 'feature=0' in repr(node)
    assert '5.5' in repr(node)

  def test_leaf_node_repr(self):
    """Leaf node should show value."""
    node = Node(leaf_node_value=42.5, is_leaf=True)
    assert 'Leaf' in repr(node)
    assert '42.5' in repr(node)


if __name__ == '__main__':
  pytest.main([__file__, '-v'])
