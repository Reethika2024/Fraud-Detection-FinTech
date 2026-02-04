"""
Unit tests for preprocessing module
"""
import pytest
import pandas as pd
import numpy as np
from preprocess import feature_engineering, split_data

def create_mock_data():
    """Create mock credit card data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create mock data similar to the real dataset
    data = pd.DataFrame({
        'Time': np.random.randint(0, 172800, n_samples),
        'Amount': np.random.exponential(scale=50, size=n_samples),
        'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    })
    
    # Add V1-V28 features (mock PCA components)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.randn(n_samples)
    
    return data

def test_feature_engineering():
    """Test feature engineering"""
    df = create_mock_data()
    df_features = feature_engineering(df)
    
    # Check that new features were created
    assert 'LogAmount' in df_features.columns
    assert 'Hour' in df_features.columns
    
    # Check that LogAmount is calculated correctly
    assert df_features['LogAmount'].min() >= 0
    
    # Check Hour is in valid range (0-48 hours since Time can be up to 172800 seconds)
    assert df_features['Hour'].min() >= 0
    assert df_features['Hour'].max() < 48  # 172800 seconds / 3600 = 48 hours

def test_split_data():
    """Test train/test split"""
    df = create_mock_data()
    df = feature_engineering(df)
    
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Check shapes
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    
    # Check that features don't include Class
    assert 'Class' not in X_train.columns
    assert 'Class' not in X_test.columns
    
    # Check that test size is approximately 20%
    total = len(X_train) + len(X_test)
    test_ratio = len(X_test) / total
    assert 0.15 < test_ratio < 0.25  # Allow some variance

def test_data_types():
    """Test that data types are correct"""
    df = create_mock_data()
    df = feature_engineering(df)
    
    # All features should be numeric
    assert df.select_dtypes(include=[np.number]).shape[1] == len(df.columns)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
