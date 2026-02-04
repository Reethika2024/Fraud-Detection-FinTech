"""
Project configuration
"""

import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'creditcard.csv')
MODELS_PATH = os.path.join(PROJECT_ROOT, 'models')
PLOTS_PATH = os.path.join(PROJECT_ROOT, 'plots')

# Data settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = 'Class'
AMOUNT_COLUMN = 'Amount'
TIME_COLUMN = 'Time'

# SMOTE
SAMPLING_STRATEGY = 0.5

# XGBoost params
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}

# Evaluation
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (10, 6)
