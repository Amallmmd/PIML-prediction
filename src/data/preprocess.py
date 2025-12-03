"""
Data preprocessing and cleaning functions
"""

import pandas as pd
import numpy as np
from pathlib import Path


def clean_data(df):
    """Clean raw data."""
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    # df = df.fillna(method='ffill')
    
    return df


def preprocess_data(input_path, output_path):
    """Main preprocessing pipeline."""
    # Load raw data
    df = pd.read_csv(input_path)
    
    # Clean data
    df = clean_data(df)
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == '__main__':
    preprocess_data('data/raw/input.csv', 'data/processed/output.csv')
