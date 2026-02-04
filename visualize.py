"""
Data visualization module for fraud detection
Generates exploratory data analysis (EDA) plots
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import *

def plot_class_distribution(df, save=True):
    """Plot distribution of fraud vs legitimate transactions"""
    plt.figure(figsize=FIGURE_SIZE)
    
    class_counts = df[TARGET_COLUMN].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    
    plt.bar(class_counts.index, class_counts.values, color=colors, alpha=0.8)
    plt.xlabel('Class (0=Legitimate, 1=Fraud)')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.xticks([0, 1], ['Legitimate', 'Fraud'])
    
    # Add percentage labels
    total = len(df)
    for i, count in enumerate(class_counts.values):
        percentage = (count / total) * 100
        plt.text(i, count, f'{percentage:.2f}%', ha='center', va='bottom')
    
    if save:
        os.makedirs(PLOTS_PATH, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_PATH, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        print(f"Class distribution plot saved to {PLOTS_PATH}")
    
    plt.close()

def plot_amount_distribution(df, save=True):
    """Plot distribution of transaction amounts"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot for legitimate transactions
    legitimate = df[df[TARGET_COLUMN] == 0][AMOUNT_COLUMN]
    axes[0].hist(legitimate, bins=50, color='#2ecc71', alpha=0.7)
    axes[0].set_xlabel('Amount')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Legitimate Transactions')
    axes[0].set_xlim([0, 500])
    
    # Plot for fraud transactions
    fraud = df[df[TARGET_COLUMN] == 1][AMOUNT_COLUMN]
    axes[1].hist(fraud, bins=50, color='#e74c3c', alpha=0.7)
    axes[1].set_xlabel('Amount')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Fraudulent Transactions')
    axes[1].set_xlim([0, 500])
    
    if save:
        os.makedirs(PLOTS_PATH, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_PATH, 'amount_distribution.png'), dpi=300, bbox_inches='tight')
        print(f"Amount distribution plot saved to {PLOTS_PATH}")
    
    plt.close()

def plot_time_distribution(df, save=True):
    """Plot transaction patterns over time"""
    plt.figure(figsize=(12, 5))
    
    # Convert Time to hours
    df['Hour'] = (df[TIME_COLUMN] / 3600) % 24
    
    # Plot for each class
    for class_val, color, label in [(0, '#2ecc71', 'Legitimate'), 
                                      (1, '#e74c3c', 'Fraud')]:
        subset = df[df[TARGET_COLUMN] == class_val]
        plt.hist(subset['Hour'], bins=24, alpha=0.6, color=color, label=label)
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Transactions')
    plt.title('Transaction Distribution by Hour')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save:
        os.makedirs(PLOTS_PATH, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_PATH, 'time_distribution.png'), dpi=300, bbox_inches='tight')
        print(f"Time distribution plot saved to {PLOTS_PATH}")
    
    plt.close()

def plot_correlation_heatmap(df, save=True):
    """Plot correlation heatmap of features"""
    plt.figure(figsize=(12, 10))
    
    # Select a subset of features for readability
    features_to_plot = [col for col in df.columns if col.startswith('V')][:10] + [AMOUNT_COLUMN, TARGET_COLUMN]
    
    correlation = df[features_to_plot].corr()
    
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap')
    
    if save:
        os.makedirs(PLOTS_PATH, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_PATH, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to {PLOTS_PATH}")
    
    plt.close()

def generate_all_plots(filepath=DATA_PATH):
    """Generate all EDA visualizations"""
    print("\n" + "="*60)
    print("GENERATING DATA VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Generate plots
    print("1. Class Distribution...")
    plot_class_distribution(df)
    
    print("\n2. Amount Distribution...")
    plot_amount_distribution(df)
    
    print("\n3. Time Distribution...")
    plot_time_distribution(df)
    
    print("\n4. Correlation Heatmap...")
    plot_correlation_heatmap(df)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETE!")
    print(f"Plots saved to: {PLOTS_PATH}")
    print("="*60)

if __name__ == "__main__":
    generate_all_plots()