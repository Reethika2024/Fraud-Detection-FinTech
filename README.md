# Credit Card Fraud Detection using XGBoost

![CI Pipeline](https://github.com/Reethika2024/Fraud-Detection-FinTech/actions/workflows/ci.yml/badge.svg)

A machine learning project to detect fraudulent credit card transactions. Built as part of my data science learning journey.

## Overview

This project uses XGBoost to identify fraud in credit card transactions. The main challenge here is the severe class imbalance - only 0.17% of transactions are fraudulent.

**Key Results:**
- Accuracy: 99.9%
- Precision: 95.8%
- Recall: 89.7%
- Successfully catches ~90% of fraud cases

## Dataset

Using the Kaggle Credit Card Fraud Detection dataset:
- 284,807 total transactions
- 492 fraudulent (0.17%)
- 28 anonymized features (PCA transformed)
- Time and Amount columns

Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Approach

### Data Processing
Created two new features:
- `LogAmount` - log transformation of transaction amount
- `Hour` - extracted from Time to capture daily patterns

### Handling Imbalance
Used SMOTE to oversample the minority class. Without this, the model would just predict "legitimate" for everything and still get 99.8% accuracy.

### Model
XGBoost classifier with these settings:
- 100 estimators
- Max depth of 6
- Learning rate 0.1

Chose XGBoost because it handles imbalanced data well and is fast to train.

## Project Structure
```
.
├── config.py                    # settings and parameters
├── preprocess.py               # data preprocessing
├── train_model.py              # model training
├── evaluate.py                 # metrics and evaluation
├── visualize.py                # data visualizations
├── main.py                     # run everything
├── test_preprocess.py          # unit tests
├── requirements.txt            # dependencies
└── Fraud_Detection_FinTech.ipynb  # original notebook
```

## Running the Code

Install dependencies:
```bash
pip install -r requirements.txt
```

Download the dataset and place `creditcard.csv` in the project folder.

Train the model:
```bash
python main.py --mode train
```

Or just run the notebook if you prefer.

## Visualizations

Check out the `plots/` folder for:
- Class distribution showing the imbalance
- Transaction amount patterns
- Time-based patterns
- Correlation heatmap

## Results

See [RESULTS.md](RESULTS.md) for detailed analysis and findings.

Main takeaway: The model catches most fraud (89.7% recall) while keeping false positives very low (95.8% precision). In production, you'd probably tune the threshold based on the cost of missed fraud vs annoying legitimate customers.

## Docker

If you want to run this in a container:
```bash
docker build -t fraud-detection .
docker run -v $(pwd)/models:/app/models fraud-detection
```

Or use docker-compose:
```bash
docker-compose up
```

## Testing

Run the tests:
```bash
pytest test_preprocess.py -v
```

## What I Learned

- Handling imbalanced datasets is tricky
- SMOTE helps but isn't perfect
- XGBoost is really solid for tabular data
- Feature engineering matters a lot
- Always look at confusion matrix, not just accuracy

## Future Ideas

- Try ensemble methods (combine multiple models)
- Experiment with neural networks
- Add more time-based features
- Test with online learning for real-time updates

## License

MIT License

## Author

Reethika
- GitHub: [@Reethika2024](https://github.com/Reethika2024)
