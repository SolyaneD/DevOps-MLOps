from app.model_utils import predict_text
import joblib
import os
from tempfile import TemporaryDirectory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def test_predict_text_smoke():
    vect = TfidfVectorizer()
    X = vect.fit_transform(["happy", "sad"])
    model = LogisticRegression().fit(X, ["happy_genre", "sad_genre"])
    # save to temp and load via joblib to mimic app
    with TemporaryDirectory() as d:
        import joblib, os
        joblib.dump(vect, os.path.join(d, "tfidf.joblib"))
        joblib.dump(model, os.path.join(d, "model.joblib"))
        loaded_vect = joblib.load(os.path.join(d, "tfidf.joblib"))
        loaded_model = joblib.load(os.path.join(d, "model.joblib"))
        res = predict_text("I am very happy", loaded_model, loaded_vect, top_k=1)
        assert isinstance(res, list)
        assert "genre" in res[0]
