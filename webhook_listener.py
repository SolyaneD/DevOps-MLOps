from http.server import BaseHTTPRequestHandler, HTTPServer
import subprocess
import json
import os

PORT = int(os.environ.get("WEBHOOK_PORT", 8000))
DOCKER_IMAGE = os.environ.get("DOCKER_IMAGE", "YOUR_DOCKERHUB_USER/mood2playlist:latest")
REPO_PATH = os.environ.get("REPO_PATH", os.getcwd())  # path where dvc repo exists

class Handler(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('content-length', 0))
        body = self.rfile.read(length)
        print("Received webhook:", body.decode())
        # Pull latest docker image
        subprocess.run(["docker", "pull", DOCKER_IMAGE])
        # In the repo path, run dvc pull to get latest model (requires DAGS token in .dvc/config or env)
        subprocess.run(["dvc", "pull"], cwd=REPO_PATH)
        # optionally stop existing container and run new one
        subprocess.run(["docker", "rm", "-f", "mood2playlist"], check=False)
        subprocess.run(["docker", "run", "-d", "--name", "mood2playlist", "-p", "5000:5000", DOCKER_IMAGE])
        self._set_response()
        self.wfile.write(json.dumps({"status": "pulled_and_restarted"}).encode())

def run(server_class=HTTPServer, handler_class=Handler):
    server_address = ('', PORT)
    httpd = server_class(server_address, handler_class)
    print(f"Listening webhook on port {PORT}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
