# ==========================================================
# Movie Genre Prediction System - Streamlit Application
# ==========================================================

# -------------------------------
# 1. Import Required Libraries
# -------------------------------

import streamlit as st
import pickle
import re
import string


# -------------------------------
# 2. Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Movie Genre Predictor",
    layout="centered"
)


# -------------------------------
# 3. Load Trained Model & Vectorizer
# -------------------------------

with open("genre_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


# -------------------------------
# 4. Text Preprocessing Function
# -------------------------------

def clean_text(text):
    """
    Cleans the input text by:
    - converting to lowercase
    - removing digits
    - removing punctuation
    - removing extra spaces
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


# -------------------------------
# 5. Application UI
# -------------------------------

st.title("🎬 Movie Genre Prediction System")

st.markdown("""
This application predicts the **genre of a movie** based on its **plot description**.

📌 **Instructions**:
- Enter a movie storyline in the text box below
- Click **Predict Genre**
- The predicted genre and confidence score will be displayed
""")


# -------------------------------
# 6. Example Input
# -------------------------------

st.markdown("### ✍ Example Movie Description")
st.code(
    "A young wizard discovers his magical powers and attends a school "
    "where dark forces are secretly rising."
)


# -------------------------------
# 7. User Input Section
# -------------------------------

user_input = st.text_area(
    label="📝 Enter Movie Description",
    height=200,
    placeholder="Type or paste the movie plot here..."
)


# -------------------------------
# 8. Prediction Logic
# -------------------------------

if st.button("🎯 Predict Genre"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter a movie description before predicting.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])

        predicted_genre = model.predict(vectorized_text)[0]
        prediction_probabilities = model.predict_proba(vectorized_text)[0]
        confidence_score = max(prediction_probabilities) * 100

        # -------------------------------
        # 9. Prediction Output Section
        # -------------------------------

        st.markdown("---")
        st.subheader("📌 Prediction Output")

        with st.container():
            st.write(
                "Based on the provided movie description, "
                "the system predicts the following genre:"
            )

            st.success(f"🎬 **Predicted Genre:** {predicted_genre}")
            st.info(f"📊 **Confidence Score:** {confidence_score:.2f}%")
