import sys, os, time, threading, logging
from supabase import create_client
from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
    Response,
    stream_with_context,
)
from flask_cors import CORS

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

rag_lock = threading.Lock()
reload_lock = threading.Lock()
MAX_FILE_SIZE = 25 * 1024 * 1024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log"),
    ],
)
logger = logging.getLogger("docquery")


class NoStatusFilter(logging.Filter):
    def filter(self, record):
        return "GET /api/status" not in record.getMessage()


log = logging.getLogger("werkzeug")
log.addFilter(NoStatusFilter())

app = Flask(__name__, static_folder="static")
CORS(app)


@app.before_request
def log_request():
    if request.path == "/api/status":
        return

    logger.info(
        "REQ | %s %s | IP=%s",
        request.method,
        request.path,
        request.remote_addr,
    )


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

rag = None
reload_status = {
    "indexing": False,
    "ready": False,
    "error": None,
}

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Docs")
os.makedirs(DOCS_DIR, exist_ok=True)


def load_rag():
    global rag
    with rag_lock:
        try:
            logger.info("Loading RAG module...")
            import main as rag_module

            rag = rag_module
            pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
            logger.info("RAG loaded. Found %d PDF(s) in Docs/", len(pdf_files))
            if pdf_files:
                logger.info("Indexing %d PDF(s) now...", len(pdf_files))
                threading.Thread(target=_reload_in_background, daemon=True).start()
        except Exception as e:
            logger.error("RAG failed to load", exc_info=True)


threading.Thread(target=load_rag, daemon=True).start()


def _reload_in_background():
    global rag, reload_status

    with reload_lock:
        try:
            reload_status["indexing"] = True
            reload_status["ready"] = False
            reload_status["error"] = None

            logger.info("Reloading RAG...")

            import importlib
            import main as rag_module

            importlib.reload(rag_module)

            rag = rag_module

            logger.info("RAG reload complete.")

            reload_status["ready"] = True

        except Exception as e:
            logger.exception("BACKGROUND RELOAD FAILED")

            reload_status["error"] = str(e)

        finally:
            reload_status["indexing"] = False


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


def get_docs_for_question(query: str):
    logger.info("RETRIEVAL | query=%s", query)
    q_lower = query.lower()
    is_overview = any(kw in q_lower for kw in OVERVIEW_KEYWORDS)

    if is_overview and rag.vectorstore is not None:
        initial_docs = rag.vectorstore.similarity_search(
            "introduction overview summary purpose topics covered", k=5
        )
    else:
        initial_docs = rag.retriever.invoke(query)[:5]

    if rag.reranker and len(initial_docs) > 0:
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = rag.reranker.predict(pairs)

        scored_docs = list(zip(initial_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        docs = [doc for doc, score in scored_docs[:5]]
        logger.info("RERANK | scores=%s", [f"{s:.3f}" for s in scores[:5]])
    else:
        docs = initial_docs[:5]

    return docs


def build_contextual_query(question, history):
    if not history:
        return question

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite follow-up questions into standalone queries. "
                    "Preserve intent and context. Keep it concise."
                ),
            }
        ]

        for msg in history[-4:]:
            messages.append(msg)

        messages.append(
            {
                "role": "user",
                "content": f"Rewrite this into a standalone query:\n{question}",
            }
        )

        resp = rag.client.chat.completions.create(
            model="openrouter/free",
            messages=messages,
            max_tokens=50,
            temperature=0.0,
        )

        rewritten = resp.choices[0].message.content.strip()
        return rewritten if rewritten else question

    except Exception:
        return question


def build_context(docs):
    """
    ✅ FIX 5: Label each chunk clearly so the model can cite it properly.
    Without labels, the model has nothing to reference and tends to just
    parrot sentences. With [Source N, p.X] tags, it can cite confidently.
    """
    parts = []
    sources = []

    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        filename = doc.metadata.get("source", "document")
        label = f"[Source {i}, p.{page} — {filename}]"
        parts.append(f"{label}\n{doc.page_content.strip()}")
        sources.append({"index": i, "page": page, "file": filename})

    context = "\n\n".join(parts)
    return context, sources


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def status():
    ready = rag is not None and rag.retriever is not None
    indexing = reload_status["indexing"]
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
    if reload_status["indexing"]:
        logger.warning("Upload rejected — already indexing")
        return jsonify({"error": "Already indexing, please wait."}), 429

    if "files" not in request.files:
        return jsonify({"error": "No files provided."}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files selected."}), 400

    uploaded = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            continue
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)

        if size > MAX_FILE_SIZE:
            return jsonify({"error": f"{file.filename} exceeds 25MB limit."}), 400

        save_path = os.path.join(DOCS_DIR, file.filename)

        file.save(save_path)

        with open(save_path, "rb") as f:
            supabase.storage.from_("DocQuery").upload(
                file.filename,
                f,
                {
                    "content-type": "application/pdf",
                    "upsert": "true"
                }
            )
            
        uploaded.append(file.filename)

    if not uploaded:
        return jsonify({"error": "No valid PDF files found."}), 400

    if rag is not None:
        logger.info("UPLOAD | files=%s | count=%d", uploaded, len(uploaded))
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
    if "/" in filename or "\\" in filename or ".." in filename:
        return jsonify({"error": "Invalid filename"}), 400

    path = os.path.join(DOCS_DIR, filename)
    real_path = os.path.realpath(path)
    real_docs = os.path.realpath(DOCS_DIR)

    if not real_path.startswith(real_docs):
        return jsonify({"error": "Invalid filename"}), 400

    if not os.path.exists(path):
        return jsonify({"error": "File not found."}), 404

    if rag is not None and rag.vectorstore is not None:
        rag.delete_document(path)
        logger.info("Removed vectors for: %s", filename)

    os.remove(path)
    logger.info("Deleted file: %s", filename)
    return jsonify({"message": f"Deleted {filename}", "indexing": True})


@app.route("/api/ask", methods=["POST"])
def ask():
    import traceback, json

    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    logger.info("ASK START | question=%s", question)
    history = (data or {}).get("history", [])
    MAX_WORDS = 50
    word_count = len(question.split())
    if word_count > MAX_WORDS:
        return (
            jsonify({"error": f"Query too long. Max {MAX_WORDS} words allowed."}),
            400,
        )
    if not question:
        return jsonify({"error": "Question is required."}), 400

    if rag is None:
        return (
            jsonify({"error": "Server is still loading, please wait and try again."}),
            503,
        )

    q_lower = question.lower()

    is_large_summary = any(
        x in q_lower
        for x in ["100 points", "50 points", "detailed summary", "full summary"]
    )

    if rag.retriever is None:
        if reload_status["indexing"]:
            return (
                jsonify(
                    {"error": "Still indexing your PDF. Please wait and try again."}
                ),
                503,
            )
        return (
            jsonify({"error": "No PDFs uploaded yet. Please upload a PDF first."}),
            400,
        )

    try:
        history = (data or {}).get("history", [])

        query = build_contextual_query(question, history)
        docs = get_docs_for_question(query)
        context, sources = build_context(docs)
        logger.info(
            "ASK CONTEXT | question=%s | docs=%d",
            question,
            len(docs),
        )
        logger.debug("DOC METADATA: %s", [d.metadata for d in docs])

        prompt = f"""
Answer the user's question clearly and naturally using the information below.

Rules:
- Write in a normal conversational tone
- Explain things simply and clearly
- Do not sound technical or robotic
- Do not mention documents, excerpts, context, retrieval, or internal processing
- Rephrase information naturally instead of copying
- If the answer is unclear or unavailable, say so politely
- When writing formulas, always use proper LaTeX
- Use \\[ ... \\] for block equations
- Use \\( ... \\) for inline equations

Information:
{context}

Question:
{question}

Answer:
"""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant.\n\n"
                    "Respond naturally and clearly for normal users.\n"
                    "Never sound technical, academic, or developer-oriented.\n\n"
                    "Never mention:\n"
                    "- retrieved documents\n"
                    "- document excerpts\n"
                    "- embeddings\n"
                    "- vector databases\n"
                    "- semantic search\n"
                    "- reranking\n"
                    "- context chunks\n"
                    "- system prompts\n"
                    "- internal processing\n"
                    "- retrieved context\n\n"
                    "Do not say phrases like:\n"
                    "- based on the provided context\n"
                    "- according to retrieved information\n"
                    "- the excerpts indicate\n"
                    "- the query appears\n"
                    "- the system found\n\n"
                    "If the question is unclear, meaningless, or random,\n"
                    "politely ask the user to rephrase it.\n\n"
                    "Answer in a natural conversational way.\n"
                    "Explain things clearly and simply.\n"
                    "Maintain conversation continuity."
                ),
            }
        ]

        for msg in history[-6:]:
            messages.append(msg)

        messages.append({"role": "user", "content": prompt})

        def generate():
            logger.info("STREAM START | question=%s", question)
            try:
                yield f"data: {json.dumps({'sources': sources})}\n\n"

                stream = rag.client.chat.completions.create(
                    model="openrouter/free",
                    messages=messages,
                    max_tokens=300,
                    stream=True,
                    temperature=0.3,
                )

                any_output = False

                for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta

                        content = getattr(delta, "content", None)

                        if content:
                            any_output = True
                            yield f"data: {json.dumps({'token': content})}\n\n"
                        else:
                            logger.debug("EMPTY DELTA: %s", chunk)

                    except Exception as e:
                        logger.error("Chunk parsing error", exc_info=True)

                if not any_output:
                    logger.warning("EMPTY RESPONSE from model")
                    yield f"data: {json.dumps({'token': '⚠️ Model returned empty response. Try again.'})}\n\n"

                logger.info("STREAM END | question=%s", question)

                yield "data: [DONE]\n\n"

            except Exception as e:
                logger.error("Streaming error in /api/ask", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except Exception as e:
        logger.error("Unhandled error in /api/ask", exc_info=True)
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
    if reload_status["indexing"]:
        return jsonify({"error": "Still indexing. Please wait a moment."}), 503

    try:
        docs = rag.vectorstore.similarity_search(
            "introduction overview summary purpose topics conclusions", k=5
        )
        context, _ = build_context(docs)
        logger.info("Summarize triggered")

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
                logger.error("Streaming error in /api/summarize", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except Exception as e:
        logger.error("Unhandled error in /api/summarize", exc_info=True)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)
