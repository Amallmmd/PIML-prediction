"""
Feature engineering and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def create_features(df):
    """Create new features from existing ones."""
    # Example feature engineering
    # df['new_feature'] = df['feature1'] * df['feature2']
    return df


def encode_categorical(df, columns):
    """Encode categorical variables."""
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df


def scale_features(df, columns):
    """Scale numerical features."""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


if __name__ == '__main__':
    print("Building features...")
