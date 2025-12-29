import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
nltk.download("punkt")

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [ps.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)
