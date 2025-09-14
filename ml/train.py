import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn
import os

DATA_PATH = "ml/dataset.csv"  # track with DVC
MODEL_DIR = "ml/outputs"

def load_data(path=DATA_PATH):
    return pd.read_csv(path)

def main():
    df = load_data()
    X = df["text"].astype(str)
    y = df["label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    clf = LogisticRegression(max_iter=1000)
    pipeline = make_pipeline(vectorizer, clf)

    mlflow.set_experiment("mood2playlist")
    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        # save components separately so app can load vectorizer + model individually
        os.makedirs(MODEL_DIR, exist_ok=True)
        vect = pipeline.named_steps['tfidfvectorizer']
        model = pipeline.named_steps['logisticregression']
        joblib.dump(vect, os.path.join(MODEL_DIR, "tfidf.joblib"))
        joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))

        mlflow.sklearn.log_model(pipeline, "sklearn_pipeline")
        print("accuracy:", acc)

if __name__ == "__main__":
    main()
