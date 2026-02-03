import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import load_data
from src.features.preprocessing import split_features_target

def test_data_loading():
    """Test if data loads correctly."""
    data = load_data('data/raw/diabetes.csv')
    assert len(data) > 0, "Data should not be empty"
    print("✓ Data loading test passed")

def test_feature_split():
    """Test if feature-target split works."""
    data = load_data('data/raw/diabetes.csv')
    X, y = split_features_target(data)
    assert X.shape[1] == 8, "Should have 8 features"
    assert len(y) == len(data), "Target length should match data length"
    print("✓ Feature split test passed")

if __name__ == "__main__":
    test_data_loading()
    test_feature_split()
    print("\nAll tests passed!")