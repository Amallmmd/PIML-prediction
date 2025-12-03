"""
Train machine learning models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path


def train_model(X_train, y_train, model):
    """Train a machine learning model."""
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    return accuracy


def save_model(model, filepath):
    """Save trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load trained model from disk."""
    return joblib.load(filepath)


if __name__ == '__main__':
    print("Training model...")
    # Load data
    # X, y = load_data()
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    # model = train_model(X_train, y_train, model_instance)
    
    # Evaluate
    # evaluate_model(model, X_test, y_test)
    
    # Save model
    # save_model(model, 'models/trained/model.pkl')
