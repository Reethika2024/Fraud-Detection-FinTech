"""
Feature importance visualization for fraud detection model
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from train_model import train_and_save
from config import *

def plot_feature_importance(model, X_train, save=True):
    """Plot feature importance from trained model"""
    
    # Get feature importance
    importance = model.feature_importances_
    feature_names = X_train.columns
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(20)
    
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title('Top 20 Most Important Features for Fraud Detection')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_PATH, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_PATH, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {PLOTS_PATH}")
    
    plt.close()
    
    # Print top 10
    print("\nTop 10 Most Important Features:")
    print("="*50)
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['feature']:15s}: {row['importance']:.4f}")
    print("="*50)
    
    return importance_df

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GENERATING FEATURE IMPORTANCE VISUALIZATION")
    print("="*60 + "\n")
    
    # Train model and get data
    model, X_test, y_test = train_and_save()
    
    # We need X_train for feature names
    from preprocess import preprocess_pipeline
    X_train, _, _, _ = preprocess_pipeline()
    
    # Plot importance
    importance_df = plot_feature_importance(model, X_train)
    
    print("\nFeature importance analysis complete!")
