🚨 Credit Card Fraud Detection using Machine Learning

End-to-end ML project to detect fraudulent credit card transactions on highly imbalanced data, with threshold optimization to maximize fraud recall.

📌 Project Overview

Credit card fraud detection is a real-world imbalanced classification problem, where fraudulent transactions account for <1% of all data.
Traditional accuracy-based evaluation fails in such scenarios.

This project builds industry-ready ML pipelines to:

Handle extreme class imbalance

Compare Logistic Regression vs Random Forest

Evaluate using Precision, Recall, F1, ROC-AUC, PR-AUC

Tune decision thresholds to maximize fraud detection (Recall)

Follow production-style modular code structure

🎯 Objective

Detect fraudulent credit card transactions while maximizing recall, ensuring fewer fraudulent transactions go undetected.

🧠 Key ML Concepts Applied

Imbalanced classification handling

Class-weighted learning

Feature engineering from timestamps

High-cardinality categorical feature reduction

Pipeline-based preprocessing

Threshold tuning beyond default 0.5

ROC-AUC & Precision-Recall analysis

📂 Project Structure
credit-card-fraud-detection/
│
├── src/
│   ├── data_preprocessing.py      # Data loading, cleaning, splitting
│   ├── feature_engineering.py     # Time & age feature creation
│   ├── train_logistic_regression.py
│   ├── train_random_forest.py
│   ├── threshold_tuning.py        # Decision threshold optimization
│   ├── evaluation.py              # Metrics & reports
│   └── utils.py                   # Common utilities & plots
│
├── notebooks/
│   └── EDA.ipynb                  # Exploratory analysis
│
├── data/
│   └── fraud_train.csv
│
├── requirements.txt
└── README.md

📊 Dataset

Source: Kaggle Credit Card Fraud Dataset

Target Variable: is_fraud

Class Distribution:

Legitimate: ~99.8%

Fraudulent: ~0.2%

⚙️ Feature Engineering

Transaction hour

Transaction day

Transaction month

Customer age (derived from DOB)

High-cardinality category reduction

Scaled numerical features

One-Hot encoded categorical features

🤖 Models Implemented
1️⃣ Logistic Regression

Class-weighted (class_weight='balanced')

Strong baseline for imbalanced data

Interpretable decision boundary

2️⃣ Random Forest Classifier

Ensemble learning

Handles non-linear patterns

Robust to feature interactions

📈 Evaluation Metrics

Accuracy is not reliable for imbalanced data.
This project focuses on:

Precision

Recall (Primary Metric)

F1 Score

ROC-AUC

PR-AUC

Confusion Matrix

🎯 Threshold Optimization (Core Highlight)

Instead of using the default threshold 0.5, multiple thresholds were evaluated:

Threshold range: 0.05 → 0.9

Selected threshold that maximizes Recall

Compared performance before & after optimization

📌 This reflects real industry fraud systems, where missing fraud is costlier than false alarms.

🏆 Final Results Summary
Model	Threshold	Precision	Recall	F1	ROC-AUC
Logistic Regression (Default)	0.50	✓	✓	✓	✓
Logistic Regression (Optimized)	Tuned	↑	↑↑	↑	✓
Random Forest (Default)	0.50	✓	✓	✓	✓
Random Forest (Optimized)	Tuned	↑	↑↑	↑	✓

➡ Optimized models significantly improved fraud recall

📌 Final Recommendation

Model selected based on maximum Recall after threshold tuning, making it suitable for real-world fraud detection systems where minimizing false negatives is critical.

🚀 How to Run
pip install -r requirements.txt
python main.py


(or run individual scripts from src/)

🧩 Skills Demonstrated

Machine Learning

Imbalanced Data Handling

Feature Engineering

Model Evaluation

Threshold Optimization

Scikit-Learn Pipelines

Production-style ML code organization

📣 Why This Project Matters

✅ Industry-aligned
✅ Interview-ready
✅ Recruiter-friendly
✅ Real-world ML logic
✅ Beyond “accuracy” mindset
