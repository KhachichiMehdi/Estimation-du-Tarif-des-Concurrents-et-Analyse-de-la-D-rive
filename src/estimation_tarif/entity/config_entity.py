from dataclasses import dataclass
from pathlib import Path


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
