"""
Example usage of the fraud detection pipeline
Demonstrates different ways to use the modules
"""

import sys
sys.path.append('..')  # Add parent directory to path

from preprocess import preprocess_pipeline, load_data, feature_engineering
from train_model import train_xgboost, save_model, load_model
from evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve
from visualize import generate_all_plots

print("="*70)
print("FRAUD DETECTION - USAGE EXAMPLES")
print("="*70)

# Example 1: Quick train and evaluate
print("\n" + "="*70)
print("EXAMPLE 1: Quick Train and Evaluate")
print("="*70)

print("\nStep 1: Preprocess data")
X_train, X_test, y_train, y_test = preprocess_pipeline()

print("\nStep 2: Train model")
model = train_xgboost(X_train, y_train)

print("\nStep 3: Evaluate model")
metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)

print("\nStep 4: Visualize results")
plot_confusion_matrix(y_test, y_pred, save=False)
plot_roc_curve(y_test, y_pred_proba, save=False)

# Example 2: Save and load model
print("\n" + "="*70)
print("EXAMPLE 2: Save and Load Model")
print("="*70)

print("\nSaving model...")
save_model(model, 'example_model.pkl')

print("\nLoading model...")
loaded_model = load_model('example_model.pkl')

print("\nMaking predictions with loaded model...")
predictions = loaded_model.predict(X_test[:5])
print(f"First 5 predictions: {predictions}")

# Example 3: Generate visualizations
print("\n" + "="*70)
print("EXAMPLE 3: Generate Data Visualizations")
print("="*70)

print("\nGenerating EDA plots...")
# Uncomment the line below to generate plots
# generate_all_plots()

# Example 4: Custom preprocessing
print("\n" + "="*70)
print("EXAMPLE 4: Custom Preprocessing Workflow")
print("="*70)

print("\nStep 1: Load raw data")
df = load_data()
print(f"Original shape: {df.shape}")

print("\nStep 2: Apply feature engineering")
df_processed = feature_engineering(df)
print(f"After feature engineering: {df_processed.shape}")
print(f"New features: {df_processed.columns.tolist()}")

print("\n" + "="*70)
print("EXAMPLES COMPLETE!")
print("="*70)