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

Metrics
-------
- compute_regression_metrics: MAE, RMSE, R²
- compute_classification_metrics: Accuracy, Precision, Recall, F1
"""

from src.decision_tree import DecisionTree
from src.node import Node
from src.random_forest import RandomForest
from src.gradient_boosting import GradientBoosting
from src.decision_tree_graph import DecisionTreeVisualizer, DecisionTreeGraph
from src.metrics import (
  compute_regression_metrics,
  compute_classification_metrics,
  mean_absolute_error,
  root_mean_squared_error,
  r_squared,
  accuracy,
  confusion_matrix,
)

__all__ = [
  'DecisionTree',
  'Node',
  'RandomForest',
  'GradientBoosting',
  'DecisionTreeVisualizer',
  'DecisionTreeGraph',
  'compute_regression_metrics',
  'compute_classification_metrics',
  'mean_absolute_error',
  'root_mean_squared_error',
  'r_squared',
  'accuracy',
  'confusion_matrix',
]
