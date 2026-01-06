"""
Decision Tree from Scratch

A pure Python implementation of decision trees and ensemble methods
for educational purposes. No sklearn dependencies in the core logic.

Main Classes
------------
- DecisionTree: Core decision tree for regression/classification
- RandomForest: Ensemble of trees trained on bootstrap samples
- GradientBoosting: Sequential ensemble correcting previous errors
- DecisionTreeVisualizer: Visualization of trained trees
"""

from src.decision_tree import DecisionTree
from src.node import Node
from src.random_forest import RandomForest
from src.gradient_boosting import GradientBoosting
from src.decision_tree_graph import DecisionTreeVisualizer, DecisionTreeGraph

__all__ = [
  'DecisionTree',
  'Node',
  'RandomForest',
  'GradientBoosting',
  'DecisionTreeVisualizer',
  'DecisionTreeGraph',
]
