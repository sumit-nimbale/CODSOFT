# preprocess.py

import re
import string

def clean_sentence(sentence):
    """
    This function cleans text:
    - converts to lowercase
    - removes numbers
    - removes punctuation
    """

    sentence = sentence.lower()
    sentence = re.sub(r"\d+", "", sentence)
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    sentence = " ".join(sentence.split())

    return sentence
