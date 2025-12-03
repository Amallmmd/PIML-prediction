"""
Tests for data processing modules
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.preprocess import clean_data


def test_clean_data():
    """Test data cleaning function."""
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
