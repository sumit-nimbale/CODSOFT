from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from config import TFIDF_NGRAMS, NB_ALPHA, LR_C, SVM_C

def get_model_configs():
    return {
        "Naive Bayes": {
            "estimator": Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("clf", MultinomialNB())
            ]),
            "params": {
                "tfidf__ngram_range": TFIDF_NGRAMS,
                "clf__alpha": NB_ALPHA
            }
        },

        "Logistic Regression": {
            "estimator": Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("clf", LogisticRegression(max_iter=1000))
            ]),
            "params": {
                "tfidf__ngram_range": TFIDF_NGRAMS,
                "clf__C": LR_C
            }
        },

        "Linear SVM": {
            "estimator": Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("clf", LinearSVC())
            ]),
            "params": {
                "tfidf__ngram_range": TFIDF_NGRAMS,
                "clf__C": SVM_C
            }
        }
    }
