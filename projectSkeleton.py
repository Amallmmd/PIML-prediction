"""
Data Science Project Structure Generator
This script creates a complete, standardized data science project structure.
"""

import os
from pathlib import Path


def create_project_structure(base_path="."):
    """
    Creates a comprehensive data science project structure.
    
    Args:
        base_path: Root directory where the project structure will be created
    """
    
    # Define the project structure
    structure = {
        "data": {
            "raw": {},
            "processed": {},
            "external": {},
            "interim": {}
        },
        "notebooks": {
            "exploratory": {},
            "reports": {}
        },
        "src": {
            "data": {},
            "features": {},
            "models": {},
            "visualization": {},
            "utils": {}
        },
        "models": {
            "trained": {},
            "checkpoints": {}
        },
        "reports": {
            "figures": {},
            "metrics": {}
        },
        "tests": {},
        "configs": {},
        "scripts": {},
        "docs": {},
        "logs": {}
    }
    
    # Files to create with their content
    files_content = {
        "README.md": """# Data Science Project

## Project Overview
Brief description of your project.

## Project Structure
```
├── data/               # Data directory
│   ├── raw/           # Original, immutable data
│   ├── processed/     # Cleaned, transformed data
│   ├── interim/       # Intermediate data transformations
│   └── external/      # External data sources
├── notebooks/         # Jupyter notebooks
│   ├── exploratory/   # EDA notebooks
│   └── reports/       # Final analysis notebooks
├── src/               # Source code
│   ├── data/          # Data loading and processing
│   ├── features/      # Feature engineering
│   ├── models/        # Model training and prediction
│   ├── visualization/ # Visualization scripts
│   └── utils/         # Utility functions
├── models/            # Trained models
│   ├── trained/       # Final trained models
│   └── checkpoints/   # Training checkpoints
├── reports/           # Generated reports
│   ├── figures/       # Graphics and figures
│   └── metrics/       # Model metrics and results
├── tests/             # Unit tests
├── configs/           # Configuration files
├── scripts/           # Standalone scripts
├── docs/              # Documentation
└── logs/              # Log files
```

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Add raw data to `data/raw/`
3. Run data processing scripts
4. Train models
5. Generate reports

## Requirements
- Python 3.8+
- See requirements.txt for dependencies
""",
        "requirements.txt": """# Data manipulation and analysis
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0

# Deep Learning (optional)
# tensorflow>=2.8.0
# torch>=1.10.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0

# Jupyter
jupyter>=1.0.0
ipykernel>=6.0.0

# Data validation
pandas-profiling>=3.0.0

# Experiment tracking
mlflow>=1.20.0

# Utilities
python-dotenv>=0.19.0
pyyaml>=5.4.0
tqdm>=4.62.0

# Testing
pytest>=6.2.0
pytest-cov>=3.0.0
""",
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Data
data/raw/*
data/processed/*
data/interim/*
data/external/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/interim/.gitkeep
!data/external/.gitkeep

# Models
models/trained/*
models/checkpoints/*
!models/trained/.gitkeep
!models/checkpoints/.gitkeep

# Logs
logs/*
!logs/.gitkeep
*.log

# Environment
.env
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Reports
reports/figures/*
reports/metrics/*
!reports/figures/.gitkeep
!reports/metrics/.gitkeep
""",
        "src/__init__.py": """\"\"\"Source code package for data science project.\"\"\"\n\n__version__ = '0.1.0'\n""",
        "src/data/__init__.py": """\"\"\"Data loading and processing modules.\"\"\"\n""",
        "src/data/make_dataset.py": """\"\"\"
Script to download or generate data and save to data/raw
\"\"\"

import pandas as pd
from pathlib import Path


def load_raw_data(filepath):
    \"\"\"Load raw data from file.\"\"\"
    return pd.read_csv(filepath)


def save_raw_data(data, filepath):
    \"\"\"Save raw data to file.\"\"\"
    data.to_csv(filepath, index=False)


if __name__ == '__main__':
    # Example usage
    print("Loading raw data...")
    # data = load_raw_data('data/raw/input.csv')
    # Save processed data
    # save_raw_data(data, 'data/raw/output.csv')
""",
        "src/data/preprocess.py": """\"\"\"
Data preprocessing and cleaning functions
\"\"\"

import pandas as pd
import numpy as np
from pathlib import Path


def clean_data(df):
    \"\"\"Clean raw data.\"\"\"
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    # df = df.fillna(method='ffill')
    
    return df


def preprocess_data(input_path, output_path):
    \"\"\"Main preprocessing pipeline.\"\"\"
    # Load raw data
    df = pd.read_csv(input_path)
    
    # Clean data
    df = clean_data(df)
    
    # Save processed data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")


if __name__ == '__main__':
    preprocess_data('data/raw/input.csv', 'data/processed/output.csv')
""",
        "src/features/__init__.py": """\"\"\"Feature engineering modules.\"\"\"\n""",
        "src/features/build_features.py": """\"\"\"
Feature engineering and transformation
\"\"\"

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def create_features(df):
    \"\"\"Create new features from existing ones.\"\"\"
    # Example feature engineering
    # df['new_feature'] = df['feature1'] * df['feature2']
    return df


def encode_categorical(df, columns):
    \"\"\"Encode categorical variables.\"\"\"
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df


def scale_features(df, columns):
    \"\"\"Scale numerical features.\"\"\"
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


if __name__ == '__main__':
    print("Building features...")
""",
        "src/models/__init__.py": """\"\"\"Model training and prediction modules.\"\"\"\n""",
        "src/models/train_model.py": """\"\"\"
Train machine learning models
\"\"\"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path


def train_model(X_train, y_train, model):
    \"\"\"Train a machine learning model.\"\"\"
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    \"\"\"Evaluate model performance.\"\"\"
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    return accuracy


def save_model(model, filepath):
    \"\"\"Save trained model to disk.\"\"\"
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    \"\"\"Load trained model from disk.\"\"\"
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
""",
        "src/models/predict.py": """\"\"\"
Make predictions using trained models
\"\"\"

import pandas as pd
import numpy as np
import joblib


def load_model(model_path):
    \"\"\"Load a trained model.\"\"\"
    return joblib.load(model_path)


def make_predictions(model, X):
    \"\"\"Make predictions on new data.\"\"\"
    return model.predict(X)


def predict_proba(model, X):
    \"\"\"Get prediction probabilities.\"\"\"
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
""",
        "src/visualization/__init__.py": """\"\"\"Visualization modules.\"\"\"\n""",
        "src/visualization/visualize.py": """\"\"\"
Create visualizations for data exploration and results
\"\"\"

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def plot_distribution(data, column, save_path=None):
    \"\"\"Plot distribution of a single variable.\"\"\"
    plt.figure()
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_matrix(data, save_path=None):
    \"\"\"Plot correlation matrix.\"\"\"
    plt.figure(figsize=(10, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(feature_names, importances, save_path=None):
    \"\"\"Plot feature importance.\"\"\"
    indices = np.argsort(importances)[::-1][:20]  # Top 20 features
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print("Creating visualizations...")
""",
        "src/utils/__init__.py": """\"\"\"Utility modules.\"\"\"\n""",
        "src/utils/helpers.py": """\"\"\"
Helper functions and utilities
\"\"\"

import os
import yaml
import json
from pathlib import Path
from datetime import datetime


def load_config(config_path):
    \"\"\"Load configuration from YAML file.\"\"\"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data, filepath):
    \"\"\"Save data to JSON file.\"\"\"
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(filepath):
    \"\"\"Load data from JSON file.\"\"\"
    with open(filepath, 'r') as f:
        return json.load(f)


def get_timestamp():
    \"\"\"Get current timestamp as string.\"\"\"
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_dir(directory):
    \"\"\"Create directory if it doesn't exist.\"\"\"
    Path(directory).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    pass
""",
        "configs/config.yaml": """# Project Configuration

project_name: "data_science_project"
version: "0.1.0"

# Data paths
data:
  raw: "data/raw"
  processed: "data/processed"
  interim: "data/interim"
  external: "data/external"

# Model parameters
model:
  random_state: 42
  test_size: 0.2
  cv_folds: 5

# Training parameters
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001

# Output paths
output:
  models: "models/trained"
  reports: "reports"
  logs: "logs"
""",
        "scripts/run_pipeline.py": """\"\"\"
Main pipeline script to run the entire workflow
\"\"\"

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.preprocess import preprocess_data
from features.build_features import create_features
from models.train_model import train_model, evaluate_model, save_model


def run_pipeline():
    \"\"\"Execute the complete data science pipeline.\"\"\"
    print("=" * 50)
    print("Starting Data Science Pipeline")
    print("=" * 50)
    
    # Step 1: Preprocess data
    print("\\n[1/4] Preprocessing data...")
    # preprocess_data('data/raw/input.csv', 'data/processed/clean_data.csv')
    
    # Step 2: Feature engineering
    print("\\n[2/4] Building features...")
    # features = create_features(data)
    
    # Step 3: Train model
    print("\\n[3/4] Training model...")
    # model = train_model(X_train, y_train, model_instance)
    
    # Step 4: Evaluate and save
    print("\\n[4/4] Evaluating and saving model...")
    # accuracy = evaluate_model(model, X_test, y_test)
    # save_model(model, 'models/trained/final_model.pkl')
    
    print("\\n" + "=" * 50)
    print("Pipeline completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    run_pipeline()
""",
        "tests/__init__.py": """\"\"\"Test package.\"\"\"\n""",
        "tests/test_data.py": """\"\"\"
Tests for data processing modules
\"\"\"

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.preprocess import clean_data


def test_clean_data():
    \"\"\"Test data cleaning function.\"\"\"
    # Create sample data with duplicates
    df = pd.DataFrame({
        'A': [1, 2, 2, 3],
        'B': [4, 5, 5, 6]
    })
    
    cleaned_df = clean_data(df)
    
    # Check duplicates are removed
    assert len(cleaned_df) == 3
    assert cleaned_df.duplicated().sum() == 0


if __name__ == '__main__':
    pytest.main([__file__])
""",
        "tests/test_models.py": """\"\"\"
Tests for model modules
\"\"\"

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


def test_model_training():
    \"\"\"Test basic model training.\"\"\"
    # Create dummy data
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Train model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Check predictions
    predictions = model.predict(X)
    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


if __name__ == '__main__':
    pytest.main([__file__])
""",
        ".env.example": """# Environment Variables Example
# Copy this file to .env and fill in your values

# Data sources
DATA_SOURCE_URL=https://example.com/data

# API Keys (if needed)
API_KEY=your_api_key_here

# Database (if needed)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=database_name
DB_USER=username
DB_PASSWORD=password

# MLflow tracking
MLFLOW_TRACKING_URI=http://localhost:5000

# Random seed for reproducibility
RANDOM_SEED=42
""",
        "notebooks/exploratory/01_data_exploration.ipynb": """{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Exploration\\n",
    "\\n",
    "Initial exploration of the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import seaborn as sns\\n",
    "\\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load data\\n",
    "# df = pd.read_csv('../../data/raw/data.csv')\\n",
    "# df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Basic Statistics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df.describe()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
""",
        "docs/project_documentation.md": """# Project Documentation

## Overview
Detailed documentation for the data science project.

## Data Dictionary
Description of all features and variables in the dataset.

| Feature | Type | Description |
|---------|------|-------------|
| feature1 | numeric | Description of feature1 |
| feature2 | categorical | Description of feature2 |

## Methodology

### Data Preprocessing
- Steps taken to clean the data
- Handling missing values
- Outlier detection and treatment

### Feature Engineering
- New features created
- Feature selection methods used
- Transformations applied

### Model Selection
- Models evaluated
- Evaluation metrics used
- Final model selection rationale

## Results
Summary of model performance and key findings.

## Future Work
Potential improvements and next steps.
"""
    }
    
    # Create directories
    def create_dirs(structure, parent_path):
        for name, children in structure.items():
            dir_path = Path(parent_path) / name
            dir_path.mkdir(exist_ok=True)
            print(f"Created directory: {dir_path}")
            
            # Create .gitkeep file for empty directories
            if not children:
                gitkeep = dir_path / ".gitkeep"
                gitkeep.touch()
            
            if children:
                create_dirs(children, dir_path)
    
    print("Creating project structure...")
    print("=" * 60)
    
    # Create all directories
    create_dirs(structure, base_path)
    
    # Create all files
    print("\nCreating project files...")
    print("=" * 60)
    for filepath, content in files_content.items():
        file_path = Path(base_path) / filepath
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created file: {file_path}")
    
    print("\n" + "=" * 60)
    print("✓ Project structure created successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Copy .env.example to .env and configure")
    print("3. Add your raw data to data/raw/")
    print("4. Start exploring in notebooks/exploratory/")
    print("5. Run pipeline: python scripts/run_pipeline.py")


if __name__ == "__main__":
    # Create project structure in current directory
    create_project_structure()
