"""
Experiment Runner for Decision Tree Analysis

This script runs experiments on both datasets:
1. Insurance (regression): Predict medical charges
2. Mobile Phones (classification): Predict price category

For each dataset, we train and evaluate:
- Decision Tree (from scratch)
- Random Forest (from scratch)
- Gradient Boosting (from scratch)
- XGBoost (library baseline for comparison)

We compute metrics on BOTH train and test sets to assess overfitting.

Usage:
------
python experiments/run_experiments.py

Outputs:
--------
- outputs/insurance_metrics.csv: Regression metrics (MAE, RMSE, R²)
- outputs/mobile_phones_metrics.csv: Classification metrics (accuracy, F1)
- outputs/mobile_phones_confusion_matrix.csv: Confusion matrix for classification
- Console output with analysis commentary
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

# XGBoost library baseline (for comparison with from-scratch implementations)
from xgboost import XGBRegressor, XGBClassifier

# Random seed for reproducibility
RANDOM_SEED = 42

# Our from-scratch implementations
from src.decision_tree import DecisionTree
from src.random_forest import RandomForest
from src.gradient_boosting import GradientBoosting
from src.data_utils import load_insurance_data, load_mobile_data
from src.metrics import (
  compute_regression_metrics,
  compute_classification_metrics,
  format_metrics_comparison,
  create_metrics_dataframe,
  confusion_matrix,
)


def ensure_outputs_dir() -> Path:
  """Create and return the outputs directory."""
  outputs_dir = project_root / 'outputs'
  outputs_dir.mkdir(exist_ok=True)
  return outputs_dir


# =============================================================================
# INSURANCE DATASET (REGRESSION)
# =============================================================================

def run_insurance_experiment() -> pd.DataFrame:
  """
  Run regression experiment on the insurance dataset.

  Trains Decision Tree, Random Forest, and Gradient Boosting models,
  evaluates on both train and test sets.

  Returns
  -------
  pd.DataFrame
    Metrics comparison table.
  """
  print("\n" + "=" * 70)
  print("INSURANCE DATASET - REGRESSION EXPERIMENT")
  print("=" * 70)

  # Load data
  print("\nLoading insurance data...")
  X_train, X_test, y_train, y_test = load_insurance_data()
  print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

  results = []

  # -------------------------------------------------------------------------
  # Model 1: Decision Tree (from scratch)
  # -------------------------------------------------------------------------
  print("\n--- Training Decision Tree ---")
  dt = DecisionTree(
    criterion='variance',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
  )
  dt_model = dt.train(X_train, y_train)
  print(f"Tree has {dt.node_count} nodes")

  # Predictions
  dt_train_pred = dt.predict(dt_model, X_train)
  dt_test_pred = dt.predict(dt_model, X_test)

  # Metrics
  dt_train_metrics = compute_regression_metrics(y_train.values, dt_train_pred)
  dt_test_metrics = compute_regression_metrics(y_test.values, dt_test_pred)
  results.append(format_metrics_comparison(
    dt_train_metrics, dt_test_metrics, 'Decision Tree'
  ))

  print(f"Train MAE: {dt_train_metrics['mae']:.0f}, Test MAE: {dt_test_metrics['mae']:.0f}")

  # -------------------------------------------------------------------------
  # Model 2: Random Forest (from scratch)
  # -------------------------------------------------------------------------
  print("\n--- Training Random Forest ---")
  rf = RandomForest(
    n_estimators=20,
    criterion='variance',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
    verbose=False,
  )
  rf_model = rf.train(X_train, y_train)

  # Predictions
  rf_train_pred = rf.predict(rf_model, X_train)
  rf_test_pred = rf.predict(rf_model, X_test)

  # Metrics
  rf_train_metrics = compute_regression_metrics(y_train.values, rf_train_pred)
  rf_test_metrics = compute_regression_metrics(y_test.values, rf_test_pred)
  results.append(format_metrics_comparison(
    rf_train_metrics, rf_test_metrics, 'Random Forest'
  ))

  print(f"Train MAE: {rf_train_metrics['mae']:.0f}, Test MAE: {rf_test_metrics['mae']:.0f}")

  # -------------------------------------------------------------------------
  # Model 3: Gradient Boosting (from scratch)
  # -------------------------------------------------------------------------
  print("\n--- Training Gradient Boosting ---")
  gb = GradientBoosting(
    n_estimators=20,
    learning_rate=0.2,
    criterion="squared_error",
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
    verbose=False,
  )
  gb_model = gb.train(X_train, y_train)

  # Predictions
  gb_train_pred = gb.predict(gb_model, X_train)
  gb_test_pred = gb.predict(gb_model, X_test)

  # Metrics
  gb_train_metrics = compute_regression_metrics(y_train.values, gb_train_pred)
  gb_test_metrics = compute_regression_metrics(y_test.values, gb_test_pred)
  results.append(format_metrics_comparison(
    gb_train_metrics, gb_test_metrics, 'Gradient Boosting'
  ))

  print(f"Train MAE: {gb_train_metrics['mae']:.0f}, Test MAE: {gb_test_metrics['mae']:.0f}")

  # Create results DataFrame
  df = create_metrics_dataframe(results)

  # Save results
  outputs_dir = ensure_outputs_dir()
  output_path = outputs_dir / 'insurance_metrics.csv'
  df.to_csv(output_path)
  print(f"\nResults saved to: {output_path}")

  return df


# =============================================================================
# MOBILE PHONES DATASET (CLASSIFICATION)
# =============================================================================

def run_mobile_phones_experiment() -> pd.DataFrame:
  """
  Run classification experiment on the mobile phones dataset.

  Trains Decision Tree, Random Forest, and Gradient Boosting models,
  evaluates on both train and test sets.

  Returns
  -------
  pd.DataFrame
    Metrics comparison table.
  """
  print("\n" + "=" * 70)
  print("MOBILE PHONES DATASET - CLASSIFICATION EXPERIMENT")
  print("=" * 70)

  # Load data
  print("\nLoading mobile phones data...")
  X_train, X_test, y_train, y_test = load_mobile_data()
  print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
  print(f"Number of classes: {y_train.nunique().values[0]}")

  results = []

  # -------------------------------------------------------------------------
  # Model 1: Decision Tree (from scratch) - Classification
  # -------------------------------------------------------------------------
  print("\n--- Training Decision Tree (Gini) ---")
  dt = DecisionTree(
    criterion='gini',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
  )
  dt_model = dt.train(X_train, y_train)
  print(f"Tree has {dt.node_count} nodes")

  # Predictions (already class labels for gini)
  dt_train_pred = dt.predict(dt_model, X_train)
  dt_test_pred = dt.predict(dt_model, X_test)

  # Metrics
  dt_train_metrics = compute_classification_metrics(y_train.values, dt_train_pred)
  dt_test_metrics = compute_classification_metrics(y_test.values, dt_test_pred)
  results.append(format_metrics_comparison(
    dt_train_metrics, dt_test_metrics, 'Decision Tree'
  ))

  print(f"Train Acc: {dt_train_metrics['accuracy']:.4f}, Test Acc: {dt_test_metrics['accuracy']:.4f}")

  # -------------------------------------------------------------------------
  # Model 2: Random Forest (from scratch) - Classification
  # -------------------------------------------------------------------------
  print("\n--- Training Random Forest (Gini) ---")
  rf = RandomForest(
    n_estimators=20,
    criterion='gini',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
    verbose=False,
  )
  rf_model = rf.train(X_train, y_train)

  # Predictions
  rf_train_pred = rf.predict(rf_model, X_train)
  rf_test_pred = rf.predict(rf_model, X_test)

  # Metrics
  rf_train_metrics = compute_classification_metrics(y_train.values, rf_train_pred)
  rf_test_metrics = compute_classification_metrics(y_test.values, rf_test_pred)
  results.append(format_metrics_comparison(
    rf_train_metrics, rf_test_metrics, 'Random Forest'
  ))

  print(f"Train Acc: {rf_train_metrics['accuracy']:.4f}, Test Acc: {rf_test_metrics['accuracy']:.4f}")

  # -------------------------------------------------------------------------
  # Model 3: Gradient Boosting (from scratch) - as pseudo-classification
  # Note: Our GB is designed for regression; for classification, we treat
  # the class labels as numeric and round predictions to nearest class.
  # -------------------------------------------------------------------------
  print("\n--- Training Gradient Boosting (Regression-based) ---")
  gb = GradientBoosting(
    n_estimators=20,
    learning_rate=0.2,
    criterion="squared_error",
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=2,
    verbose=False,
  )
  gb_model = gb.train(X_train, y_train)

  # Predictions - round to nearest class
  gb_train_pred_raw = gb.predict(gb_model, X_train)
  gb_test_pred_raw = gb.predict(gb_model, X_test)

  # Clip and round to valid class range [0, 3]
  n_classes = y_train.nunique().values[0]
  gb_train_pred = np.clip(np.round(gb_train_pred_raw), 0, n_classes - 1).astype(int)
  gb_test_pred = np.clip(np.round(gb_test_pred_raw), 0, n_classes - 1).astype(int)

  # Metrics
  gb_train_metrics = compute_classification_metrics(y_train.values, gb_train_pred)
  gb_test_metrics = compute_classification_metrics(y_test.values, gb_test_pred)
  results.append(format_metrics_comparison(
    gb_train_metrics, gb_test_metrics, 'Gradient Boosting'
  ))

  print(f"Train Acc: {gb_train_metrics['accuracy']:.4f}, Test Acc: {gb_test_metrics['accuracy']:.4f}")

  # Create results DataFrame
  df = create_metrics_dataframe(results)

  # Save results
  outputs_dir = ensure_outputs_dir()
  output_path = outputs_dir / 'mobile_phones_metrics.csv'
  df.to_csv(output_path)
  print(f"\nResults saved to: {output_path}")

  # Also save confusion matrix for test set (Decision Tree as example)
  cm, labels = confusion_matrix(y_test.values, dt_test_pred)
  cm_df = pd.DataFrame(cm, index=labels, columns=labels)
  cm_path = outputs_dir / 'mobile_phones_confusion_matrix.csv'
  cm_df.to_csv(cm_path)
  print(f"Confusion matrix saved to: {cm_path}")

  return df


# =============================================================================
# OVERFITTING ANALYSIS
# =============================================================================

def print_overfitting_analysis(
    insurance_df: pd.DataFrame,
    mobile_df: pd.DataFrame
) -> None:
  """
  Print analysis of overfitting based on train vs test metrics.
  """
  print("\n" + "=" * 70)
  print("OVERFITTING ANALYSIS")
  print("=" * 70)

  print("\n" + "-" * 50)
  print("INSURANCE DATASET (Regression)")
  print("-" * 50)
  print("\nMetrics Comparison (Train vs Test):")
  print(insurance_df.round(4).to_string())

  print(1243)
  print("\n### Overfitting Analysis:")
  for model in insurance_df.index:
    train_mae = insurance_df.loc[model, 'train_mae']
    test_mae = insurance_df.loc[model, 'test_mae']
    gap = train_mae - test_mae

    if gap > 0.15:
      status = "SIGNIFICANT OVERFITTING"
    elif gap > 0.05:
      status = "MODERATE OVERFITTING"
    else:
      status = "MINIMAL OVERFITTING"

    print(f"- {model}: Train MAE={train_mae:.4f}, Test MAE={test_mae:.4f}, "
          f"Gap={gap:.4f} -> {status}")

  print("\n### Bias-Variance Interpretation:")
  print("""
Decision Tree:
  - High variance, low bias model
  - Deep trees memorize training data -> large train/test gap
  - Expected to overfit most severely

Random Forest:
  - Reduces variance through averaging
  - Bootstrap sampling creates diversity
  - Should show smaller train/test gap than single tree

Gradient Boosting:
  - Sequential error correction can overfit with many trees
  - Learning rate provides regularization
  - Typically balances between tree and forest
""")

  print("\n" + "-" * 50)
  print("MOBILE PHONES DATASET (Classification)")
  print("-" * 50)
  print("\nMetrics Comparison (Train vs Test):")
  print(mobile_df.round(4).to_string())

  print("\n### Overfitting Analysis:")
  for model in mobile_df.index:
    train_acc = mobile_df.loc[model, 'train_accuracy']
    test_acc = mobile_df.loc[model, 'test_accuracy']
    gap = train_acc - test_acc

    if gap > 0.10:
      status = "SIGNIFICANT OVERFITTING"
    elif gap > 0.03:
      status = "MODERATE OVERFITTING"
    else:
      status = "MINIMAL OVERFITTING"

    print(f"- {model}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, "
          f"Gap={gap:.4f} -> {status}")


# =============================================================================
# MAIN
# =============================================================================

def main():
  """Run all experiments and print analysis."""
  print("=" * 70)
  print("DECISION TREE FROM SCRATCH - EXPERIMENTS")
  print("=" * 70)
  print("\nThis script evaluates our from-scratch implementations on:")
  print("1. Insurance dataset (regression)")
  print("2. Mobile phones dataset (classification)")
  print("\nWe compare TRAIN vs TEST metrics to assess overfitting.")

  # Run experiments
  insurance_results = run_insurance_experiment()
  mobile_results = run_mobile_phones_experiment()

  # Print analysis
  print_overfitting_analysis(insurance_results, mobile_results)

  print("\n" + "=" * 70)
  print("EXPERIMENT COMPLETE")
  print("=" * 70)
  print("\nOutput files:")
  print("- outputs/insurance_metrics.csv")
  print("- outputs/mobile_phones_metrics.csv")
  print("- outputs/mobile_phones_confusion_matrix.csv")


if __name__ == '__main__':
  main()
