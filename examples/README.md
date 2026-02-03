# 📚 Usage Examples

This folder contains practical examples demonstrating how to use the fraud detection modules.

## 🚀 Running Examples

### Run all examples:
```bash
cd examples
python example_usage.py
```

## 📖 Example Scenarios

### Example 1: Complete Pipeline
```python
from preprocess import preprocess_pipeline
from train_model import train_xgboost
from evaluate import full_evaluation

# Run complete pipeline
X_train, X_test, y_train, y_test = preprocess_pipeline()
model = train_xgboost(X_train, y_train)
metrics = full_evaluation(model, X_test, y_test)
```

### Example 2: Load Existing Model
```python
from train_model import load_model

# Load pre-trained model
model = load_model('fraud_detection_model.pkl')

# Make predictions
predictions = model.predict(X_test)
```

### Example 3: Generate Visualizations
```python
from visualize import generate_all_plots

# Generate all EDA plots
generate_all_plots('creditcard.csv')
```

### Example 4: Custom Hyperparameters
```python
import xgboost as xgb

# Define custom parameters
custom_params = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05
}

# Train with custom params
model = xgb.XGBClassifier(**custom_params)
model.fit(X_train, y_train)
```

## 🎯 Use Cases

1. **Research & Development**: Experiment with different models
2. **Production Deployment**: Load trained model for predictions
3. **Data Analysis**: Generate insights from transaction data
4. **Model Comparison**: Compare different algorithms

## 💡 Tips

- Always preprocess data before training
- Save models after training for reuse
- Use visualization to understand data patterns
- Test with small data samples first

## 📞 Need Help?

Check the main [README.md](../README.md) for more information.