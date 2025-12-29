# SMS Spam Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

A machine learning–based system for classifying SMS messages as **Spam** or **Ham**, built using classical NLP techniques and a structured, production-oriented workflow.

---

## Problem Statement

Unwanted spam messages affect user experience and may cause security risks.  
This project aims to build a **robust, interpretable, and well-evaluated** SMS spam classifier using traditional machine learning algorithms.

---

## Approach Overview

1. Text cleaning and normalization  
2. Feature extraction using **TF-IDF**
3. Model comparison across multiple classifiers
4. Hyperparameter tuning using **GridSearchCV**
5. Final model selection based on **F1-score**
6. Persistent storage of metrics and trained model

---

## Project Structure

sms-spam-detection/           
│               
├── data/             
│ ├── raw/             
│ │ └── spam.csv            
│ └── processed/            
│ └── cleaned_spam.csv             
│              
├── src/          
│ ├── preprocessing.py    
│ ├── train.py  
│ ├── evaluate.py  
│ ├── utils.py  
│ └── init.py   
│   
├── notebooks/   
│ ├── 01_data_exploration.ipynb   
│ ├── 02_text_preprocessing.ipynb    
│ ├── 03_model_training.ipynb    
│ └── 04_model_evaluation.ipynb    
│    
├── metrics/    
│ ├── classification_report.txt    
│ ├── confusion_matrix.csv    
│ └── final_model_metrics.json    
│   
├── artifacts/    
│ └── best_spam_model.pkl    
│    
├── docs/    
│ └── model_decisions.md    
│    
├── requirements.txt    
└── README.md     


---

## Data Description

- Dataset: SMS Spam Collection
- Target classes:
  - **Ham (0)** – Legitimate messages
  - **Spam (1)** – Unwanted messages
- Dataset is **imbalanced**, making accuracy an unreliable standalone metric

---

## Text Preprocessing

The following preprocessing steps are applied:

- Lowercasing text
- Removing digits
- Removing punctuation
- Trimming extra whitespace

These steps reduce noise and improve generalization for short-form SMS text.

---

## Feature Engineering

- **TF-IDF Vectorization**
- English stop-word removal
- N-gram tuning during hyperparameter optimization

---

## Model Comparison (Baseline)

Three models are trained and evaluated using the same preprocessing pipeline:

| Model | Strengths | Limitations |
|-----|----------|-------------|
| Naive Bayes | Fast, strong baseline for text | Lower recall on spam |
| Logistic Regression | Interpretable, stable | Limited margin separation |
| Support Vector Machine | Strong decision boundary | Higher computational cost |

### Baseline Results Summary

| Model | Accuracy | Precision | Recall | F1-score |
|------|---------|----------|-------|---------|
| Naive Bayes | ~96% | High | Moderate | Lower |
| Logistic Regression | ~96% | High | Moderate | Lower |
| SVM | ~98% | High | High | Best |

*(Exact metrics are stored in the `metrics/` directory)*

---

## Hyperparameter Tuning with GridSearchCV

The **SVM model** is further optimized using `GridSearchCV`.

### Tuned Parameters
- `tfidf__ngram_range`
- `tfidf__max_df`
- `model__C`

### GridSearch Configuration
- Cross-validation: 5-fold
- Scoring metric: **F1-score**
- Parallel execution enabled

The best estimator is selected based on cross-validated F1-score.

---

## Final Model Performance

| Metric | Score |
|------|------|
| Accuracy | ~98% |
| Precision | ~97% |
| Recall | ~89% |
| F1-score | ~93% |

Detailed artifacts:
- `classification_report.txt`
- `confusion_matrix.csv`
- `final_model_metrics.json`

---

## Confusion Matrix

![Confusion Matrix](docs/images/confusion_matrix.png)

*(Generated during model evaluation)*

---

## Inference Example

```python
prediction, probability = predict_spam(
    "Congratulations! You have won a free lottery ticket"
)


