# Contributing to Credit Card Fraud Detection

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git installed on your machine
- Basic knowledge of machine learning and fraud detection

### Setup Development Environment

1. **Fork and clone the repository:**
```bash
   git clone https://github.com/Reethika2024/Fraud-Detection-FinTech.git
   cd Fraud-Detection-FinTech
```

2. **Install dependencies:**
```bash
   pip install -r requirements.txt
```

3. **Download the dataset:**
   - Get it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Place `creditcard.csv` in the project root

## 📝 How to Contribute

### Reporting Bugs
- Use GitHub Issues
- Describe the bug clearly
- Include steps to reproduce
- Mention your Python version and OS

### Suggesting Enhancements
- Open an issue with your suggestion
- Explain why this enhancement would be useful
- Provide examples if possible

### Code Contributions

1. **Create a new branch:**
```bash
   git checkout -b feature/your-feature-name
```

2. **Make your changes:**
   - Follow PEP 8 style guidelines
   - Add docstrings to functions
   - Keep functions focused and modular

3. **Test your changes:**
```bash
   python main.py --mode train
```

4. **Commit your changes:**
```bash
   git add .
   git commit -m "Add: description of your changes"
```

5. **Push and create Pull Request:**
```bash
   git push origin feature/your-feature-name
```

## 🏗️ Project Structure
```
Fraud-Detection-FinTech/
├── config.py              # Configuration settings
├── preprocess.py          # Data preprocessing
├── train_model.py         # Model training
├── evaluate.py            # Model evaluation
├── main.py                # Main pipeline
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

## ✅ Commit Message Guidelines

Use clear, descriptive commit messages:
- `Add: new feature or file`
- `Fix: bug fix`
- `Update: improvements to existing code`
- `Refactor: code restructuring`
- `Docs: documentation changes`

## 🧪 Testing

Before submitting a PR:
- Test the preprocessing pipeline
- Ensure model trains successfully
- Verify evaluation metrics are generated
- Check for any Python errors

## 📊 Code Quality

- Write clean, readable code
- Add comments for complex logic
- Use meaningful variable names
- Follow existing code style

## 🤝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow

## 📧 Questions?

Open an issue or reach out to [@Reethika2024](https://github.com/Reethika2024)

---

Thank you for contributing! 🎉