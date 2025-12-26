# Task 2 — Credit Card Fraud Detection (Machine Learning)

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Problem](https://img.shields.io/badge/Problem-Classification-success)
![Logistic Regression](https://img.shields.io/badge/Logistic%20Regression-Classification-blue)
![Random Forest](https://img.shields.io/badge/Random%20Forest-Ensemble-green)
![Evaluation](https://img.shields.io/badge/Evaluation-Precision--Recall-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)
![Internship](https://img.shields.io/badge/Internship-CodSoft-lightgrey)

---

## Project Overview
This project implements machine learning models to detect fraudulent credit card transactions.  
The focus is on **handling highly imbalanced data** and using **appropriate evaluation metrics** rather than accuracy alone.

---

## Problem Statement
Fraud detection is a **binary classification problem**:
- `0` → Legitimate transaction  
- `1` → Fraudulent transaction  

The dataset is **severely imbalanced**, making standard accuracy an unreliable performance measure.

---

## Dataset
- Anonymized credit card transaction dataset  
- Numerical features derived from transaction behavior  
- Fraudulent transactions represent a very small percentage of total samples  

> Raw data is not included due to size and privacy constraints.

---

## Approach
The project follows a standard machine learning workflow:
1. Data loading and inspection  
2. Exploratory Data Analysis (EDA)  
3. Identification of class imbalance  
4. Data preprocessing and feature scaling  
5. Training baseline and tree-based models  
6. Evaluation using imbalance-aware metrics  

---

## Models Implemented
- **Logistic Regression** — baseline linear classifier  
- **Random Forest** — ensemble-based tree model  

These models are used to compare linear and non-linear approaches for fraud detection.

---

## Evaluation Metrics
Due to class imbalance, model performance is evaluated using:
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  
- Precision–Recall Curve  

Accuracy is not used as the primary evaluation metric.

---

## Results & Visualizations

### Class Distribution
<p align="center">
  <img src="reports/figures/class_distribution.png" width="600">
</p>

### Confusion Matrix
<p align="center">
  <img src="reports/figures/confusion_matrix.png" width="600">
</p>

### Precision–Recall Curve
<p align="center">
  <img src="reports/figures/precision_recall_curve.png" width="600">
</p>

---

