import joblib
from src.preprocessing import clean_text

def save_model(model, path: str):
    joblib.dump(model, path)


def load_model(path: str):
    return joblib.load(path)


def predict_spam(message: str, model):
    """
    Predict spam for a single message.
    """
    message = clean_text(message)
    prediction = model.predict([message])[0]
    probability = model.predict_proba([message])[0][1]
    return prediction, probability
