# predict.py

import pickle
from preprocess import clean_sentence

print("\nLoading model...")

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("Model loaded successfully!")

while True:
    text = input("\nEnter movie description (or type 'exit'): ")

    if text.lower() == "exit":
        print("Goodbye 👋")
        break

    cleaned = clean_sentence(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)

    print("Predicted Genre:", prediction[0].upper())
