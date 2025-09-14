FROM python:3.10-slim

WORKDIR /app
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app code
COPY app /app

# default model path inside container
ENV MODEL_PATH=/app/model/model.joblib
ENV VECT_PATH=/app/model/tfidf.joblib
ENV PORT=5000

EXPOSE 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "2"]
