"""
Main pipeline script to run the entire workflow
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.preprocess import preprocess_data
from features.build_features import create_features
from models.train_model import train_model, evaluate_model, save_model


def run_pipeline():
    """Execute the complete data science pipeline."""
    print("=" * 50)
    print("Starting Data Science Pipeline")
    print("=" * 50)
    
    # Step 1: Preprocess data
    print("\n[1/4] Preprocessing data...")
    # preprocess_data('data/raw/input.csv', 'data/processed/clean_data.csv')
    
    # Step 2: Feature engineering
    print("\n[2/4] Building features...")
    # features = create_features(data)
    
    # Step 3: Train model
    print("\n[3/4] Training model...")
    # model = train_model(X_train, y_train, model_instance)
    
    # Step 4: Evaluate and save
    print("\n[4/4] Evaluating and saving model...")
    # accuracy = evaluate_model(model, X_test, y_test)
    # save_model(model, 'models/trained/final_model.pkl')
    
    print("\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    run_pipeline()
