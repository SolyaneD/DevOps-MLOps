from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route("/deploy", methods=["POST"])
def deploy():
    try:
        data = request.get_json()
        image = data.get("image")
        if not image:
            return jsonify({"error": "No image provided"}), 400

        # Télécharger le modèle depuis MLflow
        model_dir = os.path.join(os.path.dirname(__file__), "model")
        os.makedirs(model_dir, exist_ok=True)
        subprocess.run([
            "mlflow", "artifacts", "download",
            "--artifact-uri", "runs:/<64d25cc093514ba7992cf44b553e01c5>/model",
            "--dst-path", model_dir
        ], check=True)

        # Pull de la dernière image Docker
        subprocess.run(["docker", "pull", image], check=True)

        # Stopper le conteneur existant (s'il existe)
        subprocess.run(["docker", "stop", "mlops-api"], check=False)
        subprocess.run(["docker", "rm", "mlops-api"], check=False)

        # Lancer le conteneur
        subprocess.run([
            "docker", "run", "-d",
            "--name", "mlops-api",
            "-p", "8000:8000",
            "-v", f"{model_dir}:/app/model",
            image
        ], check=True)

        return jsonify({"status": "deployed"}), 200

    except subprocess.CalledProcessError as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000)
