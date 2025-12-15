# ===============================
# IMPORT REQUIRED LIBRARIES
# ===============================

import re
import string
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score


# ===============================
# STEP 1: LOAD DATASET
# ===============================

print("\n==========================================")
print("STEP 1: LOADING DATASET")
print("==========================================")

df = pd.read_csv(
    r"C:\Users\sumit\OneDrive\Desktop\Task_1\Task 1 Github\main_dataset\train_data.csv",
    sep=":::",
    engine="python",
    names=["id", "title", "genre", "text"]
)

print("Dataset Loaded Successfully.")
print(f"Total records found: {len(df)}\n")


# ===============================
# STEP 2: DATA CLEANING
# ===============================

df = df[["text", "genre"]]
df.dropna(inplace=True)

df["text"] = df["text"].str.strip()
df["genre"] = df["genre"].str.strip()


# ===============================
# STEP 3: TEXT PREPROCESSING
# ===============================

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"\d+", "", sentence)
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    sentence = " ".join(sentence.split())
    return sentence

df["clean_text"] = df["text"].apply(clean_sentence)


# ===============================
# STEP 4: TRAIN-TEST SPLIT
# ===============================

X = df["clean_text"]
y = df["genre"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


# ===============================
# STEP 5: TF-IDF VECTORIZATION
# ===============================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ===============================
# STEP 6: MODEL TRAINING
# ===============================

model = MultinomialNB()
model.fit(X_train_vec, y_train)

print("\nModel training completed successfully.")


# ===============================
# STEP 7: MODEL EVALUATION
# ===============================

y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"Accuracy Score : {accuracy:.4f}")
print(f"F1 Score       : {f1:.4f}")


# ===============================
# STEP 8: SAVE MODEL & VECTORIZER
# ===============================

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and Vectorizer saved successfully.")
