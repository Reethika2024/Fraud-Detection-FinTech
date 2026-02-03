"""
Data preprocessing module for fraud detection
Handles data loading, feature engineering, and SMOTE
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from config import *

def load_data(filepath=DATA_PATH):
    """Load the credit card dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def feature_engineering(df):
    """Create new features from existing ones"""
    print("Performing feature engineering...")
    
    # Create LogAmount feature
    df['LogAmount'] = np.log1p(df[AMOUNT_COLUMN])
    
    # Create Hour feature from Time
    df['Hour'] = (df[TIME_COLUMN] / 3600) % 24
    
    # Drop original Time and Amount columns
    df = df.drop([TIME_COLUMN, AMOUNT_COLUMN], axis=1)
    
    print("Feature engineering complete")
    return df

def split_data(df):
    """Split data into train and test sets"""
    print("Splitting data into train and test sets...")
    
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    """Apply SMOTE to handle class imbalance"""
    print("Applying SMOTE for class balancing...")
    
    smote = SMOTE(sampling_strategy=SAMPLING_STRATEGY, random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Original class distribution: {dict(pd.Series(y_train).value_counts())}")
    print(f"After SMOTE: {dict(pd.Series(y_train_resampled).value_counts())}")
    
    return X_train_resampled, y_train_resampled

def preprocess_pipeline(filepath=DATA_PATH):
    """Complete preprocessing pipeline"""
    # Load data
    df = load_data(filepath)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Apply SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test

if __name__ == "__main__":
    # Test the preprocessing pipeline
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    print("\nPreprocessing complete!")
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")