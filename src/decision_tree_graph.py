"""
Decision Tree Visualization

This module provides visualization for trained decision trees using networkx
and matplotlib. It creates a graphical representation showing:
- Decision nodes (blue): Feature name and threshold for splitting
- Leaf nodes (green): Predicted values

The visualization helps understand what the tree has learned and aids in
explaining the model to others.

Requirements:
- matplotlib
- networkx
- graphviz (system installation + pydot or pygraphviz)
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from typing import Optional
from src.node import Node


class DecisionTreeVisualizer:
  """
  Visualize a trained decision tree as a graph.

  Creates a tree diagram showing:
  - Internal nodes with their split conditions (e.g., "age <= 30")
  - Leaf nodes with their prediction values
  - Color coding: blue for decision nodes, green for leaf nodes

  Example
  -------
  >>> tree = DecisionTree(max_depth=3)
  >>> model = tree.train(X_train, y_train)
  >>> viz = DecisionTreeVisualizer()
  >>> viz.visualize(model, feature_dtypes=X_train.dtypes)
  """

  def __init__(self):
    self._graph: Optional[nx.DiGraph] = None
    self._node_count: int = 0
    self._feature_dtypes: Optional[pd.Series] = None

  def _format_node_label(self, node: Node) -> str:
    """
    Create a human-readable label for a node.

    For decision nodes: "feature_name <= threshold" or "feature_name = value"
    For leaf nodes: The predicted value (rounded to 2 decimal places)

    Parameters
    ----------
    node : Node
      The node to create a label for.

    Returns
    -------
    str
      Formatted label string.
    """
    if node.is_leaf:
      # Leaf node: show the prediction value
      if isinstance(node.leaf_node_value, (int, float, np.floating)):
        return f"{node.leaf_node_value:.2f}"
      return str(node.leaf_node_value)

    # Decision node: show the split condition
    feature_name = self._feature_dtypes.keys()[node.feature_idx]
    feature_dtype = self._feature_dtypes.values[node.feature_idx]

    if feature_dtype in [int, float, np.int64, np.float64]:
      # Numeric feature: inequality comparison
      return f"{feature_name} <= {node.threshold}"
    else:
      # Categorical feature: equality comparison
      return f"{feature_name} = {node.threshold}"

  def _add_node_to_graph(
      self,
      node: Optional[Node],
      parent_id: Optional[int] = None
  ) -> None:
    """
    Recursively add nodes to the networkx graph.

    Traverses the tree depth-first, adding each node and its connection
    to the parent.

    Parameters
    ----------
    node : Node or None
      Current node to add (None for missing children).
    parent_id : int or None
      ID of the parent node (None for root).
    """
    if node is None:
      return

    self._node_count += 1
    current_id = self._node_count

    # Add node with its label and leaf status
    label = self._format_node_label(node)
    self._graph.add_node(current_id, label=label, is_leaf=node.is_leaf)

    # Connect to parent if this isn't the root
    if parent_id is not None:
      self._graph.add_edge(parent_id, current_id)

    # Recursively add children
    self._add_node_to_graph(node.left_child_node, parent_id=current_id)
    self._add_node_to_graph(node.right_child_node, parent_id=current_id)

  def _compute_label_positions(self, base_positions: dict) -> dict:
    """
    Adjust label positions for better readability.

    Shifts labels slightly above decision nodes and below leaf nodes
    to avoid overlap with the node circles.

    Parameters
    ----------
    base_positions : dict
      Node positions from graphviz layout.

    Returns
    -------
    dict
      Adjusted positions for labels.
    """
    label_positions = {}
    for node_id, (x, y) in base_positions.items():
      if self._graph.nodes[node_id]['is_leaf']:
        # Leaf labels below the node
        label_positions[node_id] = (x, y - 8)
      else:
        # Decision labels above the node
        label_positions[node_id] = (x, y + 17)
    return label_positions

  def visualize(
      self,
      tree: Node,
      feature_dtypes: pd.Series,
      figsize: tuple = (13, 8)
  ) -> None:
    """
    Visualize the decision tree as a graph.

    Parameters
    ----------
    tree : Node
      Root node of a trained decision tree.
    feature_dtypes : pd.Series
      Data types of features (from X_train.dtypes).
      Used to format split conditions appropriately.
    figsize : tuple, default=(13, 8)
      Figure size in inches (width, height).
    """
    self._feature_dtypes = feature_dtypes
    self._graph = nx.DiGraph()
    self._node_count = 0

    # Build the graph structure
    self._add_node_to_graph(tree)

    # Get node labels
    labels = nx.get_node_attributes(self._graph, 'label')

    # Compute layout using graphviz's dot algorithm (hierarchical)
    positions = nx.nx_pydot.graphviz_layout(self._graph, prog='dot')

    # Color nodes: green for leaves, blue for decision nodes
    node_colors = [
        "lightgreen" if self._graph.nodes[node]["is_leaf"] else "lightblue"
        for node in self._graph.nodes
    ]

    # Adjust label positions for clarity
    label_positions = self._compute_label_positions(positions)

    # Create the plot
    plt.figure(figsize=figsize)
    nx.draw(
        self._graph,
        positions,
        with_labels=True,
        node_size=500,
        node_color=node_colors,
        font_size=10
    )
    nx.draw_networkx_labels(
        self._graph,
        label_positions,
        labels,
        font_size=11,
        verticalalignment='top'
    )
    plt.title("Decision Tree Structure")
    plt.tight_layout()
    plt.show()


# Backwards compatibility alias
DecisionTreeGraph = DecisionTreeVisualizer
