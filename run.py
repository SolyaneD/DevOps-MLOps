import mlflow

# Démarrer un run
with mlflow.start_run() as run:
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    
    print("Run ID:", run.info.run_id)