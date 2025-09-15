from app.model_utils import predict_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tempfile import TemporaryDirectory
import joblib, os

def test_predict_text_smoke():
    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())
    ]).fit(["happy", "sad"], ["happy_genre", "sad_genre"])

    with TemporaryDirectory() as d:
        joblib.dump(model, os.path.join(d, "model.pkl"))
        loaded_model = joblib.load(os.path.join(d, "model.pkl"))
        res = predict_text("I am very happy", loaded_model, top_k=1)
        assert isinstance(res, list)
        assert "genre" in res[0]
