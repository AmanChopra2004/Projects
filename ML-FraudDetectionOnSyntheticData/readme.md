# 🕵️ Fraud Detection on Synthetic Financial Transaction Data

This project implements an end-to-end machine learning pipeline to detect fraudulent financial transactions using a synthetic dataset. The workflow mirrors real-world fraud detection systems, with emphasis on exploratory data analysis, handling class imbalance, model comparison, and evaluation using appropriate metrics.

---

## 📌 Problem Statement

Fraud detection is a binary classification problem where fraudulent transactions represent a very small fraction of total transactions. Traditional accuracy-based evaluation is misleading in such cases.

The objective of this project is to:
- Analyze patterns associated with fraudulent behavior
- Train machine learning models to identify fraud effectively
- Evaluate models using metrics suited for highly imbalanced data

---

## 📂 Dataset

The dataset is a synthetic financial transactions dataset commonly used on Kaggle for fraud detection experiments.

### Key characteristics:
- Simulates real-world mobile money transactions
- Extremely imbalanced target variable
- Contains transaction types, amounts, and account balance information

### Target Variable
- `isFraud`
  - `1` → Fraudulent transaction
  - `0` → Legitimate transaction

---

## 🔍 Exploratory Data Analysis (EDA)

The EDA process includes:
- Inspecting dataset dimensions and feature distributions
- Analyzing class imbalance between fraud and non-fraud cases
- Studying fraud occurrence across different transaction types
- Observing relationships between transaction amounts, balances, and fraud labels

**Key observation:** Fraud cases are rare, making precision, recall, and ROC-AUC more meaningful than accuracy.

---

## 🛠 Data Preprocessing

The following preprocessing steps are applied:
- Removal of non-informative identifier columns
- Encoding of categorical features
- Feature selection based on relevance
- Stratified train-test split to preserve class imbalance
- Scaling of numerical features where required

---

## 🤖 Models Implemented

The following classification models are trained and evaluated:
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

All models are evaluated on the same train-test split to ensure fair comparison.

---

## 📏 Evaluation Metrics

Given the imbalanced nature of the dataset, models are evaluated using:
- Confusion Matrix
- Precision
- Recall
- F1-Score
- ROC-AUC Score

These metrics provide a realistic assessment of fraud detection performance.

---

## 📊 Results & Insights

- Tree-based models outperform linear models for this dataset
- Random Forest and XGBoost show improved recall for fraud detection
- High accuracy does not necessarily imply effective fraud detection
- Ensemble methods handle complex, non-linear patterns better

---

## 🧠 Key Learnings

- Class imbalance significantly impacts model evaluation
- Metric selection is critical in fraud detection problems
- Transaction type and balance-related features are highly informative
- Ensemble models are well-suited for structured financial data

---

## ▶️ How to Run

### Clone the repository
```bash
git clone https://github.com/your-username/fraud-detection-synthetic-data.git
cd fraud-detection-synthetic-data
