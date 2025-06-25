"""
Data ingestion component for the estimation_tarif package.
"""

from pathlib import Path

import pandas as pd

from estimation_tarif import CustomException, logging
from estimation_tarif.apply_transformation import read_yaml
from estimation_tarif.entity import DataIngestionConfig


class DataIngestion:
    """
    Data ingestion component for the estimation_tarif package.
    This component is responsible for loading datasets from specified paths.
    """

    def __init__(self, config_path: str):
        """
        Initialize DataIngestion with a YAML config file path.
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        try:
            config = read_yaml(Path(config_path))
            self.config = DataIngestionConfig(
                train_data_path=Path(config.data_paths.train),
                test_data_path=Path(config.data_paths.test),
                drift_data_path=Path(config.data_paths.drift),
            )
            logging.info("Configuration file read successfully.")
        except Exception as e:
            logging.error("Failed to initialize DataIngestionConfig.")
            raise CustomException(e) from e

    def ingest(self):
        """
        Loads train, test, and drift datasets as pandas DataFrames.
        Returns:
            tuple: (train_df, test_df, drift_df)
        """
        try:
            train_df = pd.read_csv(self.config.train_data_path)
            logging.info(f"Train data shape: {train_df.shape}")
            test_df = pd.read_csv(self.config.test_data_path)
            logging.info(f"Test data shape: {test_df.shape}")
            drift_df = pd.read_csv(self.config.drift_data_path)
            logging.info(f"Drift data shape: {drift_df.shape}")
            logging.info("All datasets loaded successfully.")
            return train_df, test_df, drift_df
        except Exception as e:
            logging.error("Error loading datasets.")
            raise CustomException(e) from e
