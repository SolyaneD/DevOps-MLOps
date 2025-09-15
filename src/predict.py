from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model_utils import load_model, predict_text
import os

# Charger le modèle
model = load_model()

# Créer l'app
app = FastAPI()

# Activer CORS pour le front
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve index.html
@app.get("/")
def home():
    html_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
    return FileResponse(html_path)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    res = predict_text(input.text, model, top_k=1)
    return {"prediction": res[0]["genre"]}
