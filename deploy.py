from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route("/deploy", methods=["POST"])
def deploy():
    data = request.get_json()
    image = data.get("image")
    run_id = data.get("run_id")

    if not image or not run_id:
        return jsonify({"error": "Provide image and run_id"}), 400

    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)

    # Télécharger le modèle depuis MLflow
    subprocess.run([
        "mlflow", "artifacts", "download",
        "--artifact-uri", f"runs:/{run_id}/model",
        "--dst-path", model_dir
    ], check=True)

    # Pull et run l'image Docker
    subprocess.run(["docker", "pull", image], check=True)
    subprocess.run(["docker", "stop", "mlops-api"], check=False)
    subprocess.run(["docker", "rm", "mlops-api"], check=False)
    subprocess.run([
        "docker", "run", "-d",
        "--name", "mlops-api",
        "-p", "8000:8000",
        "-v", f"{model_dir}:/app/model",
        image
    ], check=True)

    return jsonify({"status": "deployed"}), 200

if __name__ == "__main__":
    app.run(port=8000)
