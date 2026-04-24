import sys, os, time, threading, logging
from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    Response,
    stream_with_context,
)
from flask_cors import CORS


class NoStatusFilter(logging.Filter):
    def filter(self, record):
        return "GET /api/status" not in record.getMessage()


log = logging.getLogger("werkzeug")
log.addFilter(NoStatusFilter())

app = Flask(__name__, static_folder="static")
CORS(app)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

rag = None
reload_status = {"running": False, "error": None}

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Docs")
os.makedirs(DOCS_DIR, exist_ok=True)


def load_rag():
    global rag
    try:
        print("Loading RAG...")
        import main as rag_module

        rag = rag_module
        print("✓ RAG loaded successfully")

        pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
        if pdf_files:
            print(f"Found {len(pdf_files)} PDF(s) in Docs/, indexing now...")
            threading.Thread(target=_reload_in_background, daemon=True).start()

    except Exception as e:
        print(f"✗ RAG failed to load: {e}")
        import traceback

        traceback.print_exc()


threading.Thread(target=load_rag, daemon=True).start()


def _reload_in_background():
    global reload_status
    reload_status["running"] = True
    reload_status["error"] = None
    try:
        rag.reload()
        print("✓ Background reload complete")
    except Exception as e:
        print(f"✗ Background reload failed: {e}")
        reload_status["error"] = str(e)
    finally:
        reload_status["running"] = False


OVERVIEW_KEYWORDS = {
    "about",
    "overview",
    "summary",
    "summarize",
    "summarise",
    "what is this",
    "what does this",
    "describe",
    "explain this",
    "tell me about",
    "give me an overview",
    "what topics",
    "what does it cover",
    "main topic",
    "purpose of",
    "what kind",
}


def get_docs_for_question(question: str):
    q_lower = question.lower()
    is_overview = any(kw in q_lower for kw in OVERVIEW_KEYWORDS)

    if is_overview and rag.vectorstore is not None:
        docs = rag.vectorstore.similarity_search(
            "introduction overview summary purpose topics covered", k=4
        )
    else:
        docs = rag.retriever.invoke(question)[:3]

    return docs


def build_context(docs):
    context = ""
    sources = []
    seen = set()
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)
        content = doc.page_content[:800]
        context += (
            f"[Source {i+1}: {os.path.basename(source)}, Page {page+1}]\n{content}\n\n"
        )
        key = (os.path.basename(source), page + 1)
        if key not in seen:
            seen.add(key)
            sources.append({"file": os.path.basename(source), "page": page + 1})
    return context, sources


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def status():
    ready = rag is not None and rag.retriever is not None
    indexing = reload_status["running"]
    error = reload_status["error"]

    if error:
        msg = f"Indexing failed: {error}"
    elif indexing:
        msg = "Indexing your PDF, please wait..."
    elif ready:
        msg = "RAG pipeline ready."
    else:
        msg = "Server still loading, please wait..."

    return jsonify(
        {"ready": ready, "indexing": indexing, "error": error, "message": msg}
    )


@app.route("/api/upload", methods=["POST"])
def upload():
    if reload_status["running"]:
        return jsonify({"error": "Already indexing, please wait."}), 429

    if "files" not in request.files:
        return jsonify({"error": "No files provided."}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files selected."}), 400

    uploaded = []
    for file in files:
        if file.filename.endswith(".pdf"):
            save_path = os.path.join(DOCS_DIR, file.filename)
            file.save(save_path)
            uploaded.append(file.filename)

    if not uploaded:
        return jsonify({"error": "No valid PDF files found."}), 400

    if rag is not None:
        threading.Thread(target=_reload_in_background, daemon=True).start()
        indexing = True
        msg = f"Uploaded {', '.join(uploaded)}. Indexing in background."
    else:
        indexing = False
        msg = f"Uploaded {', '.join(uploaded)}. Server still warming up — upload again in ~30s to index."

    return jsonify({"message": msg, "files": uploaded, "indexing": indexing})


@app.route("/api/files", methods=["GET"])
def list_files():
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    return jsonify({"files": files})


@app.route("/api/files/<filename>", methods=["DELETE"])
def delete_file(filename):
    path = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found."}), 404
    os.remove(path)
    threading.Thread(target=_reload_in_background, daemon=True).start()
    return jsonify({"message": f"Deleted {filename}", "indexing": True})


@app.route("/api/ask", methods=["POST"])
def ask():
    import traceback, json

    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400

    if rag is None:
        return (
            jsonify({"error": "Server is still loading, please wait and try again."}),
            503,
        )

    if rag.retriever is None:
        if reload_status["running"]:
            return (
                jsonify(
                    {
                        "error": "Still indexing your PDF. Please wait a moment and try again."
                    }
                ),
                503,
            )
        return (
            jsonify({"error": "No PDFs uploaded yet. Please upload a PDF first."}),
            400,
        )

    try:
        docs = get_docs_for_question(question)
        context, sources = build_context(docs)

        prompt = f"""Answer using ONLY the context below. Be concise (2-4 sentences max unless the question requires more). Mention page numbers where relevant. If not found, say so.

Context:
{context}

Question: {question}
Answer:"""

        messages = [
            {
                "role": "system",
                "content": "You are a concise document assistant. Answer only from context. Be brief and direct.",
            },
            {"role": "user", "content": prompt},
        ]

        def generate():
            try:
                yield f"data: {json.dumps({'sources': sources})}\n\n"

                stream = rag.client.chat.completions.create(
                    model="openrouter/free",
                    messages=messages,
                    max_tokens=300,
                    stream=True,
                )

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield f"data: {json.dumps({'token': delta.content})}\n\n"

                yield "data: [DONE]\n\n"

            except Exception as e:
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/summarize", methods=["POST"])
def summarize_all():
    import traceback, json

    if rag is None:
        return jsonify({"error": "Server is still loading, please wait."}), 503
    if rag.retriever is None:
        return (
            jsonify({"error": "No PDFs uploaded yet. Please upload a PDF first."}),
            400,
        )
    if reload_status["running"]:
        return jsonify({"error": "Still indexing. Please wait a moment."}), 503

    try:
        docs = rag.vectorstore.similarity_search(
            "introduction overview summary purpose topics conclusions", k=5
        )
        context, _ = build_context(docs)

        prompt = f"""Summarize the document(s) below in 2-3 short paragraphs. Cover the main purpose and key topics. Be concise.

Context:
{context}

Summary:"""

        messages = [
            {
                "role": "system",
                "content": "You are a concise document assistant. Write short, clear summaries. Never use knowledge outside the provided context.",
            },
            {"role": "user", "content": prompt},
        ]

        def generate():
            try:
                stream = rag.client.chat.completions.create(
                    model="openrouter/free",
                    messages=messages,
                    max_tokens=400,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        yield f"data: {json.dumps({'token': delta.content})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                traceback.print_exc()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
