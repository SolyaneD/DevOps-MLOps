from flask import Flask, request, jsonify
from model_utils import load_model, predict_text
import os

app = Flask(__name__)
model, vectorizer = load_model()

@app.route("/")
def ping():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "send JSON {\"text\": \"...\"}"}), 400
    text = data["text"]
    preds = predict_text(text, model, vectorizer)
    return jsonify({"predictions": preds})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
