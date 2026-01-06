"""
Data Utilities for Decision Tree Project

This module provides utilities for downloading and loading datasets used
in the decision tree examples. Datasets are downloaded from Kaggle and
stored in the data/ directory.

Requirements:
- kagglehub: For downloading Kaggle datasets

Usage:
------
>>> from src.data_utils import load_insurance_data, load_mobile_data
>>> X_train, X_test, y_train, y_test = load_insurance_data()
"""

import shutil
from pathlib import Path
from typing import Tuple
import pandas as pd


def _get_project_root() -> Path:
  """Get the project root directory."""
  # Assumes this file is in src/
  return Path(__file__).parent.parent


def _ensure_data_directory() -> Path:
  """Create and return the data directory path."""
  data_dir = _get_project_root() / 'data'
  data_dir.mkdir(exist_ok=True)
  return data_dir


def download_kaggle_dataset(kaggle_path: str, local_folder_name: str) -> Path:
  """
  Download a dataset from Kaggle to the local data directory.

  Uses kagglehub to handle authentication and downloading. You'll need
  to have Kaggle credentials configured (see kagglehub documentation).

  Parameters
  ----------
  kaggle_path : str
    The Kaggle dataset path (e.g., "username/dataset-name").
  local_folder_name : str
    Name of the folder within data/ to store the dataset.

  Returns
  -------
  Path
    Path to the local dataset directory.
  """
  try:
    import kagglehub
  except ImportError:
    raise ImportError(
        "kagglehub is required for downloading datasets. "
        "Install it with: pip install kagglehub"
    )

  data_dir = _ensure_data_directory()
  local_path = data_dir / local_folder_name
  local_path.mkdir(exist_ok=True)

  # Download the dataset
  downloaded_path = Path(kagglehub.dataset_download(kaggle_path))

  # Copy files to local data directory
  for item in downloaded_path.iterdir():
    dest = local_path / item.name
    if item.is_dir():
      shutil.copytree(item, dest, dirs_exist_ok=True)
    else:
      shutil.copy2(item, dest)

  return local_path


def train_test_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
  Split a dataframe into train and test sets.

  Simple implementation without sklearn dependency.

  Parameters
  ----------
  df : pd.DataFrame
    The full dataset.
  target_column : str
    Name of the target column.
  test_size : float, default=0.2
    Fraction of data to use for testing.
  random_state : int, default=42
    Random seed for reproducibility.

  Returns
  -------
  tuple
    (X_train, X_test, y_train, y_test)
  """
  # Shuffle the data
  df_shuffled = df.sample(frac=1, random_state=random_state)

  # Calculate split point
  split_idx = int(len(df_shuffled) * (1 - test_size))

  # Split into train and test
  train_df = df_shuffled[:split_idx]
  test_df = df_shuffled[split_idx:]

  # Separate features and target
  y_train = pd.DataFrame(train_df[target_column])
  X_train = train_df.drop(columns=[target_column])
  y_test = pd.DataFrame(test_df[target_column])
  X_test = test_df.drop(columns=[target_column])

  return X_train, X_test, y_train, y_test


def load_insurance_data(
    download_if_missing: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
  Load the healthcare insurance dataset.

  This is a REGRESSION dataset predicting insurance charges based on
  patient demographics and health factors.

  Features:
  - age: Age of the patient
  - sex: Gender (male/female)
  - bmi: Body mass index
  - children: Number of children
  - smoker: Smoking status (yes/no)
  - region: US region (northeast, northwest, southeast, southwest)

  Target:
  - charges: Medical insurance charges (continuous)

  Parameters
  ----------
  download_if_missing : bool, default=True
    If True, download the dataset from Kaggle if not present locally.

  Returns
  -------
  tuple
    (X_train, X_test, y_train, y_test) split 80/20.
  """
  data_dir = _ensure_data_directory()
  insurance_path = data_dir / 'insurance' / 'insurance.csv'

  if not insurance_path.exists():
    if download_if_missing:
      print("Downloading insurance dataset from Kaggle...")
      download_kaggle_dataset(
          "willianoliveiragibin/healthcare-insurance",
          "insurance"
      )
    else:
      raise FileNotFoundError(
          f"Dataset not found at {insurance_path}. "
          "Set download_if_missing=True to download automatically."
      )

  df = pd.read_csv(insurance_path)
  return train_test_split(df, target_column='charges')


def load_mobile_data(
    download_if_missing: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """
  Load the mobile phone price classification dataset.

  This is a CLASSIFICATION dataset predicting price range based on
  phone specifications.

  Features include:
  - battery_power, clock_speed, ram, etc.
  - Various phone hardware specifications

  Target:
  - price_range: Price category (0, 1, 2, 3)

  Parameters
  ----------
  download_if_missing : bool, default=True
    If True, download the dataset from Kaggle if not present locally.

  Returns
  -------
  tuple
    (X_train, X_test, y_train, y_test) split 80/20.
  """
  data_dir = _ensure_data_directory()
  mobile_path = data_dir / 'mobile_phones' / 'train.csv'

  if not mobile_path.exists():
    if download_if_missing:
      print("Downloading mobile phones dataset from Kaggle...")
      download_kaggle_dataset(
          "iabhishekofficial/mobile-price-classification",
          "mobile_phones"
      )
    else:
      raise FileNotFoundError(
          f"Dataset not found at {mobile_path}. "
          "Set download_if_missing=True to download automatically."
      )

  df = pd.read_csv(mobile_path)
  return train_test_split(df, target_column='price_range')
