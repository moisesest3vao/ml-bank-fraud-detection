import os
import json

# Configuration file attributes
TRAINING_FILE_PATH_CONFIG_ATTR = 'training_file_path'
X_CONFIG_ATTR = 'X'
y_CONFIG_ATTR = 'y'
NUMERICAL_FEATURES_CONFIG_ATTR = 'numerical_features'
CATEGORICAL_COLUMNS_CONFIG_ATTR = 'categorical_columns'

# Configuration file cache
CONFIG = None

def initialize_config_file():
    global CONFIG
    current_dir = os.path.dirname(__file__)
    config_path = os.path.join(current_dir, '..', 'data', 'data_config.json')

    # Open the JSON file for reading
    with open(config_path, 'r') as config_file:
        # Load the content of the JSON file into CONFIG
        CONFIG = json.load(config_file)

def get_config():
    if CONFIG is None:
        raise RuntimeError("Configuration has not been initialized.")
    return CONFIG

def get_X():
    if CONFIG is None:
        raise RuntimeError("Configuration has not been initialized.")
    return CONFIG.get(X_CONFIG_ATTR)

def get_categorical_columns():
    if CONFIG is None:
        raise RuntimeError("Configuration has not been initialized.")
    return CONFIG.get(CATEGORICAL_COLUMNS_CONFIG_ATTR)

def get_numerical_features():
    if CONFIG is None:
        raise RuntimeError("Configuration has not been initialized.")
    return CONFIG.get(NUMERICAL_FEATURES_CONFIG_ATTR)

def get_y():
    if CONFIG is None:
        raise RuntimeError("Configuration has not been initialized.")
    return CONFIG.get(y_CONFIG_ATTR)

def get_training_csv_path():
    if CONFIG is None:
        raise RuntimeError("Configuration has not been initialized.")
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, '..', 'data', CONFIG[TRAINING_FILE_PATH_CONFIG_ATTR])
