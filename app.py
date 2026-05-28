import sys, os, logging
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, jsonify
from flask_cors import CORS

import main as rag
from routes import api

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("docquery")

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

app.register_blueprint(api)

try:
    logger.info("Initializing RAG module...")
    rag.init_rag()
    logger.info("RAG initialized successfully")
except Exception:
    logger.error("RAG failed to initialize", exc_info=True)

@app.route("/api/health")
def health():
    return jsonify({"status": "healthy"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)