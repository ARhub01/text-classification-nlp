import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def load_and_preprocess(file_path, max_words=10000, max_len=200):
    df = pd.read_csv(file_path)
    df['review'] = df['review'].apply(clean_text)
    X = df['review'].values
    y = df['sentiment'].map({'negative': 0, 'positive': 1}).values

    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

    X_train, X_test, y_train, y_test = train_test_split(padded, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, max_words, max_len
