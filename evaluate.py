"""
Model evaluation module for fraud detection
Generates metrics, confusion matrix, and ROC curve
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve
)
import os
from config import *

def evaluate_model(model, X_test, y_test):
    """Evaluate model and return metrics"""
    print("Evaluating model...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Print metrics
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("="*50)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return metrics, y_pred, y_pred_proba

def plot_confusion_matrix(y_test, y_pred, save=True):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=FIGURE_SIZE)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save:
        os.makedirs(PLOTS_PATH, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_PATH, 'confusion_matrix.png'))
        print(f"Confusion matrix saved to {PLOTS_PATH}")
    
    plt.show()

def plot_roc_curve(y_test, y_pred_proba, save=True):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=FIGURE_SIZE)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save:
        os.makedirs(PLOTS_PATH, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_PATH, 'roc_curve.png'))
        print(f"ROC curve saved to {PLOTS_PATH}")
    
    plt.show()

def full_evaluation(model, X_test, y_test):
    """Complete evaluation pipeline"""
    # Evaluate metrics
    metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba)
    
    return metrics

if __name__ == "__main__":
    from train_model import train_and_save
    
    # Train model and evaluate
    model, X_test, y_test = train_and_save()
    full_evaluation(model, X_test, y_test)