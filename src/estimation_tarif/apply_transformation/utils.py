from pathlib import Path

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
