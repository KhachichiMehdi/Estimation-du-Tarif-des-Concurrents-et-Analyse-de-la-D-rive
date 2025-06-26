from dataclasses import dataclass
from pathlib import Path

from pandas import DataFrame


@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.

    Attributes:
        train_data_path (Path): Path to the training data file.
        test_data_path (Path): Path to the testing data file.
        drift_data_path (Path): Path to the drifted data file.
    """

    train_data_path: Path
    test_data_path: Path
    drift_data_path: Path


@dataclass
class DataPreprocessingConfig:
    """
    Configuration class for data preprocessing.

    Attributes:
        target_column (str): Name of the target column.
        categorical_features (list): List of categorical feature names.
        numerical_features (list): List of numerical feature names.
    """

    target_column: str
    data_train: DataFrame
    data_test: DataFrame
    data_drift: DataFrame
    categorical_features: list[str]
    numerical_features: list[str]
