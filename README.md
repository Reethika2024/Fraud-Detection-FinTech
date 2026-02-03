# 💳 Credit Card Fraud Detection using XGBoost

An end-to-end machine learning project to detect fraudulent credit card transactions using XGBoost classifier with 99%+ accuracy.

# 💳 Credit Card Fraud Detection using XGBoost

![CI Pipeline](https://github.com/Reethika2024/Fraud-Detection-FinTech/actions/workflows/ci.yml/badge.svg)

An end-to-end machine learning project...

## 📊 Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions
- **Features:** 30 (28 PCA-transformed features + Time + Amount)
- **Target:** Class (0 = legitimate, 1 = fraudulent)
- **Imbalance:** Only 0.17% fraudulent transactions

## 🎯 Project Overview

This project implements a robust fraud detection system using:
- **Feature Engineering:** Log transformation, time-based features
- **Handling Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique)
- **Model:** XGBoost Classifier
- **Evaluation:** Precision, Recall, F1-Score, ROC-AUC

## 🛠️ Technologies Used

- Python 3.x
- Pandas & NumPy - Data manipulation
- Scikit-learn - Machine learning pipeline
- XGBoost - Gradient boosting model
- Matplotlib & Seaborn - Visualization
- SMOTE - Handling class imbalance

## 📈 Results

- **Accuracy:** 99.9%
- **Precision:** High precision in detecting fraud
- **Recall:** Effective at catching fraudulent transactions
- **ROC-AUC Score:** Near-perfect classification

## 🚀 How to Run

1. **Clone the repository:**
```bash
   git clone https://github.com/Reethika2024/Fraud-Detection-FinTech.git
   cd Fraud-Detection-FinTech
```

2. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

3. **Download the dataset:**
   - Download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Place `creditcard.csv` in the project root directory

4. **Run the notebook:**
```bash
   jupyter notebook Fraud_Detection_FinTech.ipynb
```

## 📁 Project Structure
```
Fraud-Detection-FinTech/
├── Fraud_Detection_FinTech.ipynb    # Main notebook
├── creditcard.csv                   # Dataset (download separately)
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
└── LICENSE                          # MIT License
```

## 🔍 Key Features

- **Feature Engineering:** Created `LogAmount` and `Hour` features
- **Data Preprocessing:** Handled missing values and scaling
- **Class Imbalance:** Applied SMOTE to balance the dataset
- **Model Training:** XGBoost with optimized hyperparameters
- **Comprehensive Evaluation:** Confusion matrix, ROC curve, classification report

## 🐳 Docker Deployment

### Build and run with Docker:
```bash
# Build the Docker image
docker build -t fraud-detection .

# Run the container
docker run -v $(pwd)/models:/app/models -v $(pwd)/plots:/app/plots fraud-detection
```

### Or use Docker Compose:
```bash
# Build and run
docker-compose up

# Run in detached mode
docker-compose up -d

# Stop the container
docker-compose down
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Reethika**
- GitHub: [@Reethika2024](https://github.com/Reethika2024)

## 🙏 Acknowledgments

- Dataset provided by [Machine Learning Group - ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Inspired by real-world FinTech fraud detection systems