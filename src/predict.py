from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from app.model_utils import load_model, predict_text

model = load_model()

# API FastAPI
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    res = predict_text(input.text, model, top_k=1)
    return {"prediction": res[0]["genre"]}