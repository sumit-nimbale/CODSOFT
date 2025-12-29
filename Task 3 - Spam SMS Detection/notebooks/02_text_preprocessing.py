import pandas as pd
import numpy as np 
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_csv("spam.csv",encoding = "latin-1" )
print(df.info())

df = df[['v1', 'v2']]
df.rename(columns={'v1':'label' , 'v2' : 'message'}, inplace = True)
df.info()

#LowerCase
df['message'] = df['message'].str.lower()

#Remove Punctuation & Numbers
df['message'] = df['message'].apply(lambda x:re.sub(r'[^a-z\s]', '', x))

#Tokenization
df['tokens'] = df['message'].apply(nltk.word_tokenize)

#Stopword removal
stop_words = set(stopwords.words('english'))
df['tokens'] = df['tokens'].apply(
    lambda x: [word for word in x if word not in stop_words]
)

#Stemming
ps = PorterStemmer()
df['tokens'] = df['tokens'].apply(
    lambda x : [ps.stem(word) for word in x]
)

#join tokens
df['clean_meaasage'] = df['tokens'].apply(lambda x:' '.join(x))
