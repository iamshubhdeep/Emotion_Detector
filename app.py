import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("emotion.csv")

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(preprocess_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['emotion']

# Model
model = MultinomialNB()
model.fit(X, y)

# Streamlit UI
st.title("AI-Powered Emotion Detection from Text")
st.write("Enter a sentence and the model will predict the emotion.")

user_input = st.text_area("Enter text here:")

if st.button("Predict Emotion"):
    clean_input = preprocess_text(user_input)
    vector_input = vectorizer.transform([clean_input])
    prediction = model.predict(vector_input)
    st.success(f"Predicted Emotion: {prediction[0]}")
