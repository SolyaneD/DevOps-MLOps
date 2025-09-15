FROM python:3.11-slim

WORKDIR /app

# Copier les dépendances et installer
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le modèle
COPY app/model/ /app/model/

# Copier le reste du code
COPY . .

# Lancer l'API
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]
