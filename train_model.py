"""
Model training module for fraud detection
Trains XGBoost classifier and saves the model
"""

import xgboost as xgb
import joblib
import os
from preprocess import preprocess_pipeline
from config import *

def train_xgboost(X_train, y_train):
    """Train XGBoost classifier"""
    print("Training XGBoost model...")
    
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)
    
    print("Model training complete!")
    return model

def save_model(model, filename='fraud_detection_model.pkl'):
    """Save trained model to disk"""
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_PATH, exist_ok=True)
    
    filepath = os.path.join(MODELS_PATH, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filename='fraud_detection_model.pkl'):
    """Load trained model from disk"""
    filepath = os.path.join(MODELS_PATH, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found at {filepath}")
    
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model

def train_and_save():
    """Complete training pipeline"""
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    
    # Train model
    model = train_xgboost(X_train, y_train)
    
    # Save model
    save_model(model)
    
    return model, X_test, y_test

if __name__ == "__main__":
    # Train and save the model
    model, X_test, y_test = train_and_save()
    print("\nTraining pipeline complete!")