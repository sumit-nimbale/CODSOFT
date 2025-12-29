# =========================
# 1. Import Required Libraries
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay


# =========================
# 2. Load Dataset
# =========================

df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df[['v1', 'v2']]

# Rename columns for clarity
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)

print(df.info())


# =========================
# 3. Feature & Target Split
# =========================

X = df['message']   # Text messages
y = df['label']     # Spam / Ham labels


# =========================
# 4. TF-IDF Vectorization
# =========================

tfidf = TfidfVectorizer(
    stop_words='english',   # Remove common English words
    max_features=3000       # Limit vocabulary size
)

# Convert text into TF-IDF matrix
X_tfidf = tfidf.fit_transform(X)

print("TF-IDF Shape:", X_tfidf.shape)

# Optional: view some feature names
print("Sample Features:", tfidf.get_feature_names_out()[:20])


# =========================
# 5. Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# =========================
# 6. MODEL 1: Multinomial Naive Bayes
# =========================

print("=" * 60)
print("MULTINOMIAL NAIVE BAYES")
print("=" * 60)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Accuracy:", nb_accuracy)

print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))


# =========================
# 7. MODEL 2: Logistic Regression
# =========================

print("=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Accuracy:", lr_accuracy)

print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))


# =========================
# 8. MODEL 3: Linear SVM
# =========================

print("=" * 60)
print("SUPPORT VECTOR MACHINE")
print("=" * 60)

svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, y_pred_svm)
print("SVM Accuracy:", svm_accuracy)

print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))


# =========================
# 9. Confusion Matrix Visualization
# =========================

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_nb)
plt.title("Naive Bayes Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lr)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm)
plt.title("Linear SVM Confusion Matrix")
plt.show()

