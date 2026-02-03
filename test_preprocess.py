"""
Unit tests for preprocessing module
Tests data loading, feature engineering, and SMOTE application
"""

import pytest
import pandas as pd
import numpy as np
from preprocess import (
    load_data, 
    feature_engineering, 
    split_data, 
    apply_smote
)

class TestPreprocessing:
    """Test suite for preprocessing functions"""
    
    def test_load_data(self):
        """Test if data loads correctly"""
        # This assumes creditcard.csv exists
        try:
            df = load_data()
            assert isinstance(df, pd.DataFrame), "Data should be a DataFrame"
            assert not df.empty, "DataFrame should not be empty"
            assert 'Class' in df.columns, "Should have 'Class' column"
            print("✓ test_load_data passed")
        except FileNotFoundError:
            pytest.skip("creditcard.csv not found - skipping test")
    
    def test_feature_engineering(self):
        """Test feature engineering creates correct features"""
        # Create sample data
        sample_data = pd.DataFrame({
            'Time': [0, 3600, 7200],
            'Amount': [100, 50, 200],
            'V1': [1.0, 2.0, 3.0],
            'Class': [0, 1, 0]
        })
        
        result = feature_engineering(sample_data.copy())
        
        # Check new features exist
        assert 'LogAmount' in result.columns, "LogAmount should be created"
        assert 'Hour' in result.columns, "Hour should be created"
        
        # Check old features removed
        assert 'Time' not in result.columns, "Time should be removed"
        assert 'Amount' not in result.columns, "Amount should be removed"
        
        # Check LogAmount calculation
        assert result['LogAmount'].iloc[0] == np.log1p(100), "LogAmount calculation error"
        
        # Check Hour calculation
        assert result['Hour'].iloc[1] == 1.0, "Hour should be 1 (3600/3600 % 24)"
        
        print("✓ test_feature_engineering passed")
    
    def test_split_data(self):
        """Test train-test split works correctly"""
        # Create sample data
        sample_data = pd.DataFrame({
            'V1': np.random.randn(100),
            'V2': np.random.randn(100),
            'Class': np.random.choice([0, 1], 100)
        })
        
        X_train, X_test, y_train, y_test = split_data(sample_data)
        
        # Check shapes
        assert len(X_train) + len(X_test) == 100, "Split should use all data"
        assert len(X_train) == len(y_train), "X_train and y_train size mismatch"
        assert len(X_test) == len(y_test), "X_test and y_test size mismatch"
        
        # Check target removed from features
        assert 'Class' not in X_train.columns, "Class should not be in X_train"
        assert 'Class' not in X_test.columns, "Class should not be in X_test"
        
        print("✓ test_split_data passed")
    
    def test_apply_smote(self):
        """Test SMOTE balancing works"""
        # Create imbalanced sample data
        X_train = pd.DataFrame({
            'V1': np.random.randn(100),
            'V2': np.random.randn(100)
        })
        # 90% class 0, 10% class 1 (imbalanced)
        y_train = pd.Series([0]*90 + [1]*10)
        
        X_resampled, y_resampled = apply_smote(X_train, y_train)
        
        # Check data was actually resampled
        assert len(X_resampled) > len(X_train), "SMOTE should increase data size"
        assert len(X_resampled) == len(y_resampled), "X and y size mismatch"
        
        # Check minority class increased
        minority_count_after = sum(y_resampled == 1)
        minority_count_before = sum(y_train == 1)
        assert minority_count_after > minority_count_before, "Minority class should increase"
        
        print("✓ test_apply_smote passed")

def run_all_tests():
    """Run all tests manually"""
    test = TestPreprocessing()
    
    print("\n" + "="*50)
    print("RUNNING PREPROCESSING TESTS")
    print("="*50 + "\n")
    
    try:
        test.test_load_data()
    except Exception as e:
        print(f"✗ test_load_data failed: {e}")
    
    try:
        test.test_feature_engineering()
    except Exception as e:
        print(f"✗ test_feature_engineering failed: {e}")
    
    try:
        test.test_split_data()
    except Exception as e:
        print(f"✗ test_split_data failed: {e}")
    
    try:
        test.test_apply_smote()
    except Exception as e:
        print(f"✗ test_apply_smote failed: {e}")
    
    print("\n" + "="*50)
    print("TESTS COMPLETE")
    print("="*50)

if __name__ == "__main__":
    run_all_tests()