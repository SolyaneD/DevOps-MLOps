import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import mlflow
import mlflow.sklearn

# Authentification DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "SolyaneD"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "0a1113b9cb6e966d05def3a46bca6644d2b9c9f1"

# Config MLflow pour DagsHub
mlflow.set_tracking_uri("https://dagshub.com/SolyaneD/DevOps-MLOps.mlflow")
mlflow.set_experiment("mood-music")

# Charger le dataset
df = pd.read_csv("data/mood_samples.csv")
X, y = df["text"], df["mood"]

# Pipeline NLP + modèle
pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

# Entraînement et tracking MLflow
with mlflow.start_run():
    pipe.fit(X, y)
    acc = pipe.score(X, y)
    mlflow.log_metric("train_accuracy", acc)
    # mlflow.sklearn.log_model(pipe, "model")

# Sauvegarder localement
joblib.dump(pipe, "model.pkl")
print(f"Training done, accuracy={acc}")
