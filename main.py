"""
Main script for Credit Card Fraud Detection
Complete end-to-end pipeline
"""

import argparse
from preprocess import preprocess_pipeline
from train_model import train_xgboost, save_model, load_model
from evaluate import full_evaluation
from config import *

def train_pipeline():
    """Run complete training pipeline"""
    print("\n" + "="*60)
    print("FRAUD DETECTION - TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Step 1: Preprocess data
    print("Step 1: Data Preprocessing")
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    
    # Step 2: Train model
    print("\nStep 2: Model Training")
    model = train_xgboost(X_train, y_train)
    
    # Step 3: Save model
    print("\nStep 3: Saving Model")
    save_model(model)
    
    # Step 4: Evaluate model
    print("\nStep 4: Model Evaluation")
    metrics = full_evaluation(model, X_test, y_test)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    
    return model, metrics

def predict_pipeline(model_path='fraud_detection_model.pkl'):
    """Run prediction pipeline on test data"""
    print("\n" + "="*60)
    print("FRAUD DETECTION - PREDICTION PIPELINE")
    print("="*60 + "\n")
    
    # Load model
    print("Loading trained model...")
    model = load_model(model_path)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    
    # Evaluate
    print("\nEvaluating model...")
    metrics = full_evaluation(model, X_test, y_test)
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE!")
    print("="*60)
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Credit Card Fraud Detection Pipeline'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        choices=['train', 'predict'],
        help='Mode: train (train new model) or predict (use existing model)'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_pipeline()
    elif args.mode == 'predict':
        predict_pipeline()