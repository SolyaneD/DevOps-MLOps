import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/model.joblib")
VECT_PATH = os.environ.get("VECT_PATH", "/app/model/tfidf.joblib")

def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
    return model, vectorizer

def predict_text(text, model, vectorizer, top_k=3):
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    idx = probs.argsort()[::-1][:top_k]
    return [{"genre": classes[i], "score": float(probs[i])} for i in idx]
