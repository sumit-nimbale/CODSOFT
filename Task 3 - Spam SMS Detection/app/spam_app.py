import streamlit as st
import joblib
import re
import string

st.set_page_config(page_title="SMS Spam Detection", layout="centered")

st.title("SMS Spam Detection System")
st.caption("Machine Learning model using TF-IDF and Support Vector Machine (SVM)")

model = joblib.load("best_spam_model.pkl")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

st.markdown("### Enter SMS Text")

user_input = st.text_area(
    label="",
    placeholder="Type or paste the SMS message here...",
    height=120
)


if st.button("Classify Message"):
    if user_input.strip() == "":
        st.warning("Please enter an SMS message to analyze.")
    else:
        cleaned_text = clean_text(user_input)

        prediction = model.predict([cleaned_text])[0]
        probability = model.predict_proba([cleaned_text])[0][1]

        st.markdown("---")
        st.markdown("### Prediction Result")

        if prediction == 1:
            st.error(
                f"**Result:** Spam Message\n\n"
                f"**Spam Probability:** {probability:.2%}"
            )
        else:
            st.success(
                f"**Result:** Legitimate (Ham)\n\n"
                f"**Confidence:** {(1 - probability):.2%}"
            )


st.markdown("---")
st.caption("Model trained on SMS data | TF-IDF + Support Vector Machine")

