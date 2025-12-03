"""
Script to download or generate data and save to data/raw
"""

import pandas as pd
from pathlib import Path


def load_raw_data(filepath):
    """Load raw data from file."""
    return pd.read_csv(filepath)


def save_raw_data(data, filepath):
    """Save raw data to file."""
    data.to_csv(filepath, index=False)


if __name__ == '__main__':
    # Example usage
    print("Loading raw data...")
    # data = load_raw_data('data/raw/input.csv')
    # Save processed data
    # save_raw_data(data, 'data/raw/output.csv')
