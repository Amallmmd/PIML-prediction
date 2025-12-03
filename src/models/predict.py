"""
Make predictions using trained models
"""

import pandas as pd
import numpy as np
import joblib


def load_model(model_path):
    """Load a trained model."""
    return joblib.load(model_path)


def make_predictions(model, X):
    """Make predictions on new data."""
    return model.predict(X)


def predict_proba(model, X):
    """Get prediction probabilities."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    else:
        raise AttributeError("Model does not support probability predictions")


if __name__ == '__main__':
    # Load model
    # model = load_model('models/trained/model.pkl')
    
    # Load new data
    # X_new = pd.read_csv('data/processed/new_data.csv')
    
    # Make predictions
    # predictions = make_predictions(model, X_new)
    # print(predictions)
    pass
