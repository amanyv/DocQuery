import sys, os, time, threading, logging
from flask import Flask, request, jsonify, send_from_directory
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
            "introduction overview summary purpose topics covered", k=8
        )
    else:
        docs = rag.retriever.invoke(question)

    return docs


def build_context(docs):
    context = ""
    sources = []
    seen = set()
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)
        context += f"[Source {i+1}: {os.path.basename(source)}, Page {page+1}]\n{doc.page_content}\n\n"
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
    import traceback

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

        prompt = f"""You are a helpful document assistant. Answer the question below using ONLY the provided context.

Instructions:
- Write in clear, natural paragraphs. Avoid excessive bullet points or large headers unless the content truly needs structure.
- Be concise and direct. Get to the answer quickly.
- For specific questions, mention the relevant page number naturally in your answer (e.g. "According to page 3...").
- For overview/summary questions, give a well-organized but readable answer.
- If the answer is not in the context, say: "I couldn't find this information in the uploaded documents."
- Do NOT use knowledge outside the context.

Context:
{context}

Question: {question}

Answer:"""

        messages = [
            {
                "role": "system",
                "content": "You are a helpful document assistant. Write clear, natural prose. Be concise. Never use knowledge outside the provided context.",
            },
            {"role": "user", "content": prompt},
        ]

        response = rag.client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
            max_tokens=600,
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer, "sources": sources})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/summarize", methods=["POST"])
def summarize_all():
    import traceback

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
            "introduction overview summary purpose topics conclusions", k=10
        )
        context, _ = build_context(docs)

        prompt = f"""You are a helpful document assistant. Summarize the document(s) below in clear, readable paragraphs.

Instructions:
- Write naturally, as if explaining to a colleague.
- Cover the main purpose, key topics, and any important conclusions.
- If multiple documents are present, briefly address each one.
- Keep it concise — aim for 3 to 5 paragraphs total.
- Do NOT use knowledge outside the context.

Context:
{context}

Summary:"""

        messages = [
            {
                "role": "system",
                "content": "You are a helpful document assistant. Write in clear, natural paragraphs. Never use knowledge outside the provided context.",
            },
            {"role": "user", "content": prompt},
        ]

        response = rag.client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
            max_tokens=700,
        )
        answer = response.choices[0].message.content
        return jsonify({"answer": answer})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)