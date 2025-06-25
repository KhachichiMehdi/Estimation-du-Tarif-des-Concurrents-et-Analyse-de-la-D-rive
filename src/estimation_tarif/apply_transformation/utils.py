import os
from pathlib import Path

import pandas as pd
import yaml
from box import ConfigBox  # type: ignore
from box.exceptions import BoxValueError  # type: ignore

from estimation_tarif import CustomException, logging


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads yaml file and returns the content as a ConfigBox.

    Args:
        path_to_yaml (Path): path-like input.

    Raises:
        ValueError: if yaml file is empty.
        CustomException: If any other error occurs during loading.

    Returns:
        ConfigBox: Loaded content as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if not content:
                raise BoxValueError("YAML file is empty")
            logging.info(f"YAML file loaded successfully from {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError as e:
        logging.error(f"YAML file is empty: {path_to_yaml}")
        raise CustomException(e) from e
    except yaml.YAMLError as e:
        logging.error(f"YAML parsing error in file: {path_to_yaml}")
        raise CustomException(e) from e
    except Exception as e:
        logging.exception(
            f"Unexpected error occurred while loading YAML file: {path_to_yaml}"
        )
        raise CustomException(e) from e


def create_directories(paths: list, verbose=True):
    """
    Creates directories if they do not exist.

    Args:
        paths (list): List of directory paths to create.
        verbose (bool): If True, logs the creation or existence of directories.
    """
    try:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                if verbose:
                    logging.info(f"Directory created: {path}")
            else:
                if verbose:
                    logging.info(f"Directory already exists: {path}")
    except Exception as e:
        logging.error(f"Error occurred while creating directories: {e}")
        raise CustomException(e) from e


def load_data(path: Path) -> pd.DataFrame:
    """
    Loads data from a CSV file into a pandas DataFrame.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.

    Raises:
        CustomException: If there is an error loading the data.
    """
    try:
        df = pd.read_csv(path)
        logging.info(f"Data loaded successfully from {path}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {path}: {e}")
        raise CustomException(e) from e
