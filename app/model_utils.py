import joblib
import os

# Ici on force le path local, plus besoin de MLflow pour la démo
MODEL_PATH = os.environ.get("MODEL_PATH", "app/model/model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model

def predict_text(text, model, top_k=3):
    probs = model.predict_proba([text])[0]
    classes = model.classes_
    idx = probs.argsort()[::-1][:top_k]
    return [{"genre": classes[i], "score": float(probs[i])} for i in idx]
