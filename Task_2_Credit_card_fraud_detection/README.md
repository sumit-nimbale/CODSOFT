# 💳 Credit Card Fraud Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Imbalanced Data](https://img.shields.io/badge/Data-Imbalanced-red)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-Classification-blue)
![Random Forest](https://img.shields.io/badge/Random%20Forest-Ensemble-green)


---

## 🚀 Project Overview

This project focuses on **detecting fraudulent credit card transactions** using machine learning techniques on **highly imbalanced data**.  
Special emphasis is placed on **Recall maximization**, **threshold tuning**, and **robust evaluation metrics**.

---

## 🎯 Objective

- Detect fraudulent transactions (minority class)
- Handle extreme class imbalance
- Optimize decision thresholds to **maximize Recall**
- Compare models using multiple evaluation metrics

---

## 🧠 Models Implemented

| Model | Description |
|-----|------------|
| Logistic Regression | Baseline linear model with class weighting |
| Random Forest | Ensemble model capturing non-linear patterns |

---

## ⚖️ Class Imbalance

- Fraud transactions: **~0.17%**
- Legitimate transactions: **~99.83%**

📉 Highly skewed dataset → Accuracy alone is misleading.

**Visualization:**
## Class Distribution
![Class Distribution](reports/figures/class_distribution.png)


---

## 🔧 Feature Engineering

- Transaction hour
- Transaction day
- Transaction month
- Customer age
- Categorical encoding (One-Hot)
- Numerical scaling (StandardScaler)

---

## 📊 Evaluation Metrics Used

- Precision
- Recall (primary focus)
- F1-Score
- ROC-AUC
- PR-AUC

---

## 🧪 Logistic Regression Results

### Confusion Matrix
![LR Confusion Matrix](reports/figures/logistic_regression_confusion_matrix.png)

### Precision-Recall Curve
![PR Curve](reports/figures/precision_recall_curve.png)

---

## 🌲 Random Forest Results

### Confusion Matrix
![RF Confusion Matrix](reports/figures/random_forest_cm.png)

---

## 🎯 Threshold Optimization

Instead of using the default **0.5 threshold**, we tuned thresholds to **maximize Recall**.

### Threshold Tuning Visualization
![Threshold Tuning](reports/figures/thresholding_tuning.png)

### Optimized Thresholds
| Model | Optimal Threshold |
|-----|------------------|
| Logistic Regression | Optimized |
| Random Forest | Optimized |

---

## 🏆 Final Model Comparison

| Model | Precision | Recall | F1 | ROC-AUC |
|-----|---------|-------|----|--------|
| Logistic Regression (Optimized) | High | Improved | Balanced | Strong |
| Random Forest (Optimized) | Moderate | **Highest Recall** | Strong | Best |

📌 **Final Recommendation:**  
➡️ **Random Forest with optimized threshold** for fraud detection.

---

## 📁 Repository Structure

