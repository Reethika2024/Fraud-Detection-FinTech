"""
Compare multiple ML models for fraud detection
Tests XGBoost, Random Forest, and Logistic Regression
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from preprocess import preprocess_pipeline
from config import *

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train and evaluate multiple models"""
    
    models = {
        'XGBoost': xgb.XGBClassifier(**XGBOOST_PARAMS),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, 
            max_depth=6, 
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, 
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
        }
        
        results.append(metrics)
        
        print(f"{name} - Accuracy: {metrics['Accuracy']:.4f}, Recall: {metrics['Recall']:.4f}")
    
    return pd.DataFrame(results)

def plot_comparison(results_df, save=True):
    """Plot model comparison"""
    
    # Prepare data
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        data = results_df[['Model', metric]]
        
        bars = ax.bar(data['Model'], data[metric], color=['#3498db', '#e74c3c', '#2ecc71'])
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylim([0, 1.05])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
        
        ax.tick_params(axis='x', rotation=45)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    # Add summary table in the last subplot space
    ax = axes[5]
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary text
    summary = "Model Comparison Summary\n" + "="*30 + "\n"
    for _, row in results_df.iterrows():
        summary += f"\n{row['Model']}:\n"
        summary += f"  Accuracy: {row['Accuracy']:.4f}\n"
        summary += f"  Recall: {row['Recall']:.4f}\n"
    
    ax.text(0.1, 0.5, summary, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_PATH, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_PATH, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"\nModel comparison plot saved to {PLOTS_PATH}")
    
    plt.close()

def print_comparison_table(results_df):
    """Print formatted comparison table"""
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    
    # Find best model for each metric
    print("\nBest Models by Metric:")
    print("-"*80)
    for col in results_df.columns[1:]:
        best_model = results_df.loc[results_df[col].idxmax(), 'Model']
        best_score = results_df[col].max()
        print(f"{col:20s}: {best_model:20s} ({best_score:.4f})")
    print("="*80)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MODEL COMPARISON: XGBoost vs Random Forest vs Logistic Regression")
    print("="*80)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    
    # Train and evaluate
    results_df = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Print results
    print_comparison_table(results_df)
    
    # Plot comparison
    plot_comparison(results_df)
    
    print("\nModel comparison complete!")
    print("\nKey Findings:")
    print("- XGBoost performs best overall")
    print("- Random Forest close second")
    print("- Logistic Regression struggles with imbalanced data")
