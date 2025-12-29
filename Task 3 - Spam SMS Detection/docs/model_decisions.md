# Model Design Decisions
SMS Spam Detection System

---

## 1. Problem Definition

The objective of this project is to design a supervised machine learning system
that classifies SMS messages as **Spam (1)** or **Ham (0)** with high reliability.

### Operational Context
- Spam messages negatively impact user trust and may cause financial harm.
- The cost of **false negatives (missed spam)** is higher than false positives.
- Therefore, model evaluation prioritizes **Recall** and **F1-score** over Accuracy.

---

## 2. Dataset Summary

- Dataset: SMS Spam Collection
- Total samples: ~5,500
- Target distribution:
  - Ham (majority class)
  - Spam (minority class)

### Key Implication
The dataset is **class-imbalanced**, making Accuracy an unsuitable standalone
metric. Metrics robust to imbalance are required.

---

## 3. Text Preprocessing Strategy

### Applied Steps
1. Lowercasing
2. Removal of non-alphabetic characters
3. Tokenization
4. Stopword removal
5. Stemming using Porter Stemmer

### Rationale
- Reduces vocabulary size and noise
- Improves generalization for linear models
- Computationally efficient for classical ML pipelines

Lemmatization was intentionally avoided due to:
- Additional computational overhead
- Dependence on part-of-speech tagging
- Marginal benefit for short SMS text

---

## 4. Feature Representation

### TF-IDF Vectorization

TF-IDF was selected as the feature representation method.

#### Configurations Evaluated
- Unigrams `(1,1)`
- Unigrams + Bigrams `(1,2)`

### Justification
- Down-weights frequent but uninformative terms
- Highlights discriminative spam-related tokens
- Produces sparse, well-conditioned inputs for linear classifiers

Alternative approaches (e.g., word embeddings, deep encoders) were excluded due
to dataset size and deployment simplicity requirements.

---

## 5. Model Candidates

The following models were evaluated:

| Model | Reason for Inclusion |
|------|----------------------|
| Multinomial Naive Bayes | Strong baseline for text data |
| Logistic Regression | Probabilistic output and interpretability |
| Linear SVM | Robust margin-based classifier |

All models were trained using the same TF-IDF pipeline to ensure fair comparison.

---

## 6. Hyperparameter Optimization

Hyperparameter tuning was performed using **GridSearchCV** with:
- 5-fold cross-validation
- F1-score as the optimization metric

### Tuned Parameters
- TF-IDF n-gram range
- Smoothing parameter (Naive Bayes)
- Regularization strength `C` (Logistic Regression, SVM)

This ensured systematic and reproducible model selection.

---

## 7. Evaluation Metrics

### Primary Metrics
- Recall
- Precision
- F1-score

### Secondary Metrics
- ROC-AUC
- Precision–Recall Curve

### Metric Selection Rationale
- Recall captures the ability to detect spam
- F1-score balances recall and precision
- PR-AUC provides better insight under class imbalance

---

## 8. Decision Threshold Optimization

For Logistic Regression, the default probability threshold (0.5) was not assumed
to be optimal.

### Approach
- Thresholds were evaluated in the range 0.1–0.9
- F1-score was computed at each threshold

### Result
Threshold tuning improved recall while maintaining acceptable precision,
resulting in better operational performance.

---

## 9. Final Model Selection

### Selected Model
**Logistic Regression with TF-IDF (1,2)**

### Reasons for Selection
- Strong and consistent F1-score
- Supports probability-based decisions
- Enables threshold tuning
- Easier deployment compared to SVM
- Interpretable coefficients

---

## 10. Limitations and Trade-offs

### Current Limitations
- Bag-of-words representation lacks semantic understanding
- No explicit handling of URLs, emojis, or special tokens
- English-only preprocessing

### Planned Improvements
- Character n-grams for robustness
- Class-weighted loss functions
- Transformer-based models (e.g., BERT) for semantic modeling
- Continuous model retraining pipeline

---

## 11. Production Readiness

The system is designed with production considerations:
- Modular pipeline architecture
- Reusable preprocessing and inference logic
- Model persistence and versioning
- Compatibility with API and UI deployment

---

## 12. Summary

This project emphasizes **clarity, robustness, and deployability**.
All design choices were guided by dataset characteristics, business constraints,
and industry-standard machine learning practices.
