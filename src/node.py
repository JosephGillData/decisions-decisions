"""
Node class for Decision Tree

A decision tree is composed of nodes. There are two types:
1. INTERNAL NODES (decision nodes): Test a feature against a threshold and
   route samples to left or right child based on the result.
2. LEAF NODES: Store a prediction value (no children).

This separation mirrors how the tree makes predictions:
- At internal nodes: "Is feature X <= threshold? Go left if yes, right if no."
- At leaf nodes: "Return this prediction value."
"""

from __future__ import annotations
from typing import Optional


class Node:
  """
  A node in the decision tree.

  Nodes can be either:
  - Internal (decision) nodes: Have feature_idx, threshold, and children
  - Leaf nodes: Have leaf_node_value and is_leaf=True

  The split semantics for internal nodes:
  - Left child: samples where feature[feature_idx] <= threshold
  - Right child: samples where feature[feature_idx] > threshold

  Attributes
  ----------
  feature_idx : int or None
    For internal nodes: the index of the feature used for splitting.
    For leaf nodes: None.

  threshold : float or None
    For internal nodes: the threshold value for the split.
    Samples with feature_value <= threshold go left, otherwise right.
    For leaf nodes: None.

  left_child_node : Node or None
    The left child node (samples where feature <= threshold).

  right_child_node : Node or None
    The right child node (samples where feature > threshold).

  leaf_node_value : float or None
    For leaf nodes: the prediction value (mean of targets that reached this leaf).
    For internal nodes: None.

  is_leaf : bool
    True if this is a leaf node, False if it's an internal decision node.

  n_samples : int or None
    Number of training samples that reached this node (useful for debugging).

  Example
  -------
  Creating an internal node:
  >>> node = Node(feature_idx=2, threshold=5.5, n_samples=100)
  >>> node.left_child_node = Node(leaf_node_value=3.2, is_leaf=True)
  >>> node.right_child_node = Node(leaf_node_value=7.8, is_leaf=True)

  Creating a leaf node:
  >>> leaf = Node(leaf_node_value=42.0, is_leaf=True)
  """

  def __init__(
      self,
      feature_idx: Optional[int] = None,
      threshold: Optional[float] = None,
      left_child_node: Optional[Node] = None,
      right_child_node: Optional[Node] = None,
      leaf_node_value: Optional[float] = None,
      is_leaf: bool = False,
      n_samples: Optional[int] = None,
  ):
    # Internal node attributes (used for making split decisions)
    self.feature_idx = feature_idx
    self.threshold = threshold
    self.left_child_node = left_child_node
    self.right_child_node = right_child_node

    # Leaf node attributes (used for making predictions)
    self.leaf_node_value = leaf_node_value
    self.is_leaf = is_leaf

    # Metadata (useful for visualization and debugging)
    self.n_samples = n_samples

  def __repr__(self) -> str:
    """String representation for debugging."""
    if self.is_leaf:
      return f"Leaf(value={self.leaf_node_value:.2f})"
    else:
      return f"Node(feature={self.feature_idx}, threshold={self.threshold})"
