"""
Evaluation Metrics Module

Provides metrics for evaluating regression and classification models.
Implemented from scratch without sklearn to maintain the educational focus.

Regression Metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (Coefficient of Determination)

Classification Metrics:
- Accuracy
- Precision, Recall, F1 (per-class and macro)
- Confusion Matrix
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Union, List


# =============================================================================
# REGRESSION METRICS
# =============================================================================

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """
  Calculate Mean Absolute Error (MAE).

  MAE = (1/n) * sum(|y_true - y_pred|)

  MAE is easy to interpret: it's the average prediction error in the
  same units as the target variable.

  Parameters
  ----------
  y_true : np.ndarray
    Ground truth values.
  y_pred : np.ndarray
    Predicted values.

  Returns
  -------
  float
    Mean absolute error (lower is better).
  """
  y_true = np.asarray(y_true).ravel()
  y_pred = np.asarray(y_pred).ravel()
  return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """
  Calculate Root Mean Squared Error (RMSE).

  RMSE = sqrt((1/n) * sum((y_true - y_pred)^2))

  RMSE penalizes large errors more heavily than MAE due to squaring.
  Useful when large errors are particularly undesirable.

  Parameters
  ----------
  y_true : np.ndarray
    Ground truth values.
  y_pred : np.ndarray
    Predicted values.

  Returns
  -------
  float
    Root mean squared error (lower is better).
  """
  y_true = np.asarray(y_true).ravel()
  y_pred = np.asarray(y_pred).ravel()
  mse = np.mean((y_true - y_pred) ** 2)
  return np.sqrt(mse)


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """
  Calculate R² (Coefficient of Determination).

  R² = 1 - SS_res / SS_tot
     = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2)

  R² indicates what proportion of variance in the target is explained
  by the model:
  - R² = 1: Perfect predictions
  - R² = 0: Model no better than predicting the mean
  - R² < 0: Model worse than predicting the mean

  Parameters
  ----------
  y_true : np.ndarray
    Ground truth values.
  y_pred : np.ndarray
    Predicted values.

  Returns
  -------
  float
    R-squared score (higher is better, max 1.0).
  """
  y_true = np.asarray(y_true).ravel()
  y_pred = np.asarray(y_pred).ravel()

  ss_res = np.sum((y_true - y_pred) ** 2)
  ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

  if ss_tot == 0:
    return 0.0  # All targets are identical

  return 1 - (ss_res / ss_tot)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
  """
  Compute all regression metrics.

  Parameters
  ----------
  y_true : np.ndarray
    Ground truth values.
  y_pred : np.ndarray
    Predicted values.

  Returns
  -------
  dict
    Dictionary with 'mae', 'rmse', 'r2' keys.
  """
  return {
    'mae': mean_absolute_error(y_true, y_pred),
  }


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
  """
  Calculate classification accuracy.

  Accuracy = (correct predictions) / (total predictions)

  Parameters
  ----------
  y_true : np.ndarray
    Ground truth class labels.
  y_pred : np.ndarray
    Predicted class labels.

  Returns
  -------
  float
    Accuracy score between 0 and 1 (higher is better).
  """
  y_true = np.asarray(y_true).ravel()
  y_pred = np.asarray(y_pred).ravel()
  return np.mean(y_true == y_pred)


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List = None
) -> Tuple[np.ndarray, List]:
  """
  Compute confusion matrix.

  Element [i, j] is the count of samples with true label i predicted as j.

  Parameters
  ----------
  y_true : np.ndarray
    Ground truth class labels.
  y_pred : np.ndarray
    Predicted class labels.
  labels : list, optional
    List of unique labels. If None, inferred from data.

  Returns
  -------
  tuple
    (confusion_matrix, labels): The confusion matrix and the label order.
  """
  y_true = np.asarray(y_true).ravel()
  y_pred = np.asarray(y_pred).ravel()

  if labels is None:
    labels = sorted(list(set(y_true) | set(y_pred)))

  n_labels = len(labels)
  label_to_idx = {label: i for i, label in enumerate(labels)}

  cm = np.zeros((n_labels, n_labels), dtype=int)

  for true, pred in zip(y_true, y_pred):
    if true in label_to_idx and pred in label_to_idx:
      cm[label_to_idx[true], label_to_idx[pred]] += 1

  return cm, labels


def precision_recall_f1_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List = None
) -> Dict[str, Dict]:
  """
  Compute precision, recall, and F1 score for each class.

  For class c:
  - Precision = TP / (TP + FP) - "Of predicted c, how many are correct?"
  - Recall = TP / (TP + FN) - "Of actual c, how many did we find?"
  - F1 = 2 * (precision * recall) / (precision + recall)

  Parameters
  ----------
  y_true : np.ndarray
    Ground truth class labels.
  y_pred : np.ndarray
    Predicted class labels.
  labels : list, optional
    List of unique labels.

  Returns
  -------
  dict
    Dictionary mapping each label to its precision, recall, and f1.
  """
  cm, labels = confusion_matrix(y_true, y_pred, labels)

  results = {}
  for i, label in enumerate(labels):
    tp = cm[i, i]
    fp = np.sum(cm[:, i]) - tp  # Column sum minus diagonal
    fn = np.sum(cm[i, :]) - tp  # Row sum minus diagonal

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    results[label] = {
      'precision': precision,
      'recall': recall,
      'f1': f1,
      'support': int(np.sum(cm[i, :]))  # Number of true samples of this class
    }

  return results


def macro_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
  """
  Compute macro-averaged precision, recall, and F1.

  Macro averaging: compute metric for each class, then average.
  Treats all classes equally regardless of support.

  Parameters
  ----------
  y_true : np.ndarray
    Ground truth class labels.
  y_pred : np.ndarray
    Predicted class labels.

  Returns
  -------
  dict
    Dictionary with macro precision, recall, and f1.
  """
  per_class = precision_recall_f1_per_class(y_true, y_pred)

  precisions = [m['precision'] for m in per_class.values()]
  recalls = [m['recall'] for m in per_class.values()]
  f1s = [m['f1'] for m in per_class.values()]

  return {
    'precision_macro': np.mean(precisions),
    'recall_macro': np.mean(recalls),
    'f1_macro': np.mean(f1s),
  }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
  """
  Compute all classification metrics.

  Parameters
  ----------
  y_true : np.ndarray
    Ground truth class labels.
  y_pred : np.ndarray
    Predicted class labels.

  Returns
  -------
  dict
    Dictionary with accuracy and macro-averaged precision/recall/f1.
  """
  metrics = {'accuracy': accuracy(y_true, y_pred)}
  metrics.update(macro_precision_recall_f1(y_true, y_pred))
  return metrics


# =============================================================================
# RESULTS FORMATTING
# =============================================================================

def format_metrics_comparison(
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    model_name: str
) -> Dict[str, float]:
  """
  Format train and test metrics into a single dictionary for comparison.

  Parameters
  ----------
  train_metrics : dict
    Metrics computed on training set.
  test_metrics : dict
    Metrics computed on test set.
  model_name : str
    Name of the model.

  Returns
  -------
  dict
    Combined dictionary with train_ and test_ prefixed keys.
  """
  result = {'model': model_name}

  for key, value in train_metrics.items():
    result[f'train_{key}'] = value

  for key, value in test_metrics.items():
    result[f'test_{key}'] = value

  return result


def create_metrics_dataframe(
    results: List[Dict[str, float]]
) -> pd.DataFrame:
  """
  Create a DataFrame from a list of metric results.

  Parameters
  ----------
  results : list of dict
    List of dictionaries from format_metrics_comparison.

  Returns
  -------
  pd.DataFrame
    DataFrame with models as rows and metrics as columns.
  """
  df = pd.DataFrame(results)
  df = df.set_index('model')
  return df
