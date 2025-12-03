"""
Helper functions and utilities
"""

import os
import yaml
import json
from pathlib import Path
from datetime import datetime


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data, filepath):
    """Save data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_timestamp():
    """Get current timestamp as string."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    Path(directory).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    pass
