import pandas as pd
import joblib
import os
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Setup
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
MODEL_FILE = 'email_model.pkl'
VECTORIZER_FILE = 'vectorizer.pkl'

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return " ".join([word for word in text.split() if word not in stop_words])

def train_if_missing():
    if not os.path.exists(MODEL_FILE):
        print("Initial training in progress...")
        data = {
            "text": ["Urgent server down", "Job application position", "Daily horoscope", "Meeting tomorrow"] * 50,
            "label": ["high", "medium", "low", "medium"] * 50
        }
        df = pd.DataFrame(data)
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(df['text'].apply(clean_text))
        model = MultinomialNB()
        model.fit(X, df['label'])
        joblib.dump(model, MODEL_FILE)
        joblib.dump(tfidf, VECTORIZER_FILE)

if __name__ == "__main__":
    train_if_missing()
    print("Backend Engine Ready. Use app_ui.py for the Web Dashboard.")