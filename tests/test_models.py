"""
Tests for model modules
"""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


def test_model_training():
    """Test basic model training."""
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
