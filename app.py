import sys, os, threading, logging, json
from supabase import create_client
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS

import main as rag 

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = None
try:
    if SUPABASE_URL and SUPABASE_KEY:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase connected successfully")
except Exception as e:
    print("Supabase initialization failed:", e)
    supabase = None

reload_lock = threading.Lock()
MAX_FILE_SIZE = 25 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("docquery")

class NoStatusFilter(logging.Filter):
    def filter(self, record):
        return "GET /api/status" not in record.getMessage()

log = logging.getLogger("werkzeug")
log.addFilter(NoStatusFilter())

app = Flask(__name__, static_folder="static")
CORS(app)

reload_status = {"indexing": False, "ready": False, "error": None}
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Docs")
os.makedirs(DOCS_DIR, exist_ok=True)

def load_rag():
    """Initialize RAG components once at startup."""
    global reload_status
    try:
        logger.info("Initializing RAG module...")
        rag.init_rag()
        reload_status["ready"] = (rag.retriever is not None)
        logger.info("RAG initialized successfully")
    except Exception:
        logger.error("RAG failed to initialize", exc_info=True)

def _background_worker(uploaded_files_data):
    """
    10. Move Supabase upload fully to background so UI unblocks instantly.
    """
    global reload_status

    with reload_lock:
        try:
            reload_status["indexing"] = True
            reload_status["ready"] = False
            reload_status["error"] = None

            file_paths = []
            
            for file_data in uploaded_files_data:
                path = file_data["path"]
                filename = file_data["filename"]
                file_paths.append(path)

                if supabase:
                    try:
                        with open(path, "rb") as f:
                            supabase.storage.from_("DocQuery").upload(
                                filename, f, {"content-type": "application/pdf", "upsert": "true"}
                            )
                        logger.info(f"Uploaded to Supabase: {filename}")
                    except Exception as e:
                        logger.error(f"Supabase upload failed: {e}")

            logger.info("Indexing new files into Vector DB...")
            rag.add_documents(file_paths)
            
            reload_status["ready"] = (rag.retriever is not None)
            logger.info("Background indexing complete.")

        except Exception as e:
            logger.exception("BACKGROUND WORKER FAILED")
            reload_status["error"] = str(e)
        finally:
            reload_status["indexing"] = False

OVERVIEW_KEYWORDS = {"about", "overview", "summary", "summarize", "describe", "explain this", "tell me about", "what does it cover"}

def get_docs_for_question(query: str):
    logger.info("RETRIEVAL | query=%s", query)
    q_lower = query.lower()
    is_overview = any(kw in q_lower for kw in OVERVIEW_KEYWORDS)

    if is_overview and rag.vectorstore is not None:
        initial_docs = rag.vectorstore.similarity_search("introduction overview summary purpose topics covered", k=8)
    else:
        initial_docs = rag.retriever.invoke(query)[:8]

    if hasattr(rag, "reranker") and rag.reranker is not None:
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = rag.reranker.predict(pairs)

        scored_docs = list(zip(initial_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        docs = [doc for doc, score in scored_docs[:5]]
        logger.info("RERANK COMPLETE | Top score: %.3f", scores[0] if len(scores) > 0 else 0)
    else:
        docs = initial_docs[:5]

    return docs

def build_contextual_query(question, history):
    if not history:
        return question
    try:
        messages = [{"role": "system", "content": "Rewrite the user's follow-up question into a clear, standalone query based on the conversation history. Reply ONLY with the new query."}]
        for msg in history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": f"Rewrite this: {question}"})

        resp = rag.client.chat.completions.create(
            model="openai/gpt-oss-120b:free",
            messages=messages,
            max_tokens=30,
            temperature=0.0,
        )
        rewritten = resp.choices[0].message.content.strip()
        return rewritten if rewritten else question
    except Exception:
        return question

def build_context(docs):
    parts = []
    sources = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        filename = doc.metadata.get("source", "document")
        basename = os.path.basename(filename)
        label = f"[Source {i}, p.{page} - {basename}]"
        parts.append(f"{label}\n{doc.page_content.strip()}")
        sources.append({"index": i, "page": page, "file": basename})
    
    return "\n\n".join(parts), sources

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/status")
def status():
    ready = rag.retriever is not None
    indexing = reload_status["indexing"]
    error = reload_status["error"]
    msg = f"Indexing failed: {error}" if error else "Indexing PDF..." if indexing else "Pipeline ready." if ready else "Waiting..."
    return jsonify({"ready": ready, "indexing": indexing, "error": error, "message": msg})

@app.route("/api/upload", methods=["POST"])
def upload():
    if reload_status["indexing"]:
        return jsonify({"error": "Already indexing, please wait."}), 429
    if "files" not in request.files:
        return jsonify({"error": "No files provided."}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files selected."}), 400

    uploaded = []
    uploaded_files_data = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            continue
        
        filename = secure_filename(file.filename)
        save_path = os.path.join(DOCS_DIR, filename)

        if os.path.exists(save_path) and rag.vectorstore:
            rag.delete_document(save_path)
            os.remove(save_path)

        file.save(save_path)
        uploaded_files_data.append({"path": save_path, "filename": filename})
        uploaded.append(filename)

    if not uploaded:
        return jsonify({"error": "No valid PDF files found."}), 400

    threading.Thread(target=_background_worker, args=(uploaded_files_data,), daemon=True).start()

    return jsonify({"message": "Files uploaded. Indexing in background.", "files": uploaded, "indexing": True})

@app.route("/api/files", methods=["GET"])
def list_files():
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    return jsonify({"files": files})

@app.route("/api/files/<filename>", methods=["DELETE"])
def delete_file(filename):
    path = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(path):
        return jsonify({"error": "File not found."}), 404

    if rag.vectorstore is not None:
        rag.delete_document(path)
    
    os.remove(path)
    reload_status["ready"] = (rag.retriever is not None)
    return jsonify({"message": f"Deleted {filename}"})

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required."}), 400
    if rag.retriever is None:
        return jsonify({"error": "No documents available. Upload a PDF."}), 400

    try:
        history = data.get("history", [])
        
        query = build_contextual_query(question, history)
        logger.info(f"Query: {query}")

        docs = get_docs_for_question(query)
        context, sources = build_context(docs)

        prompt = f"""
You are an intelligent document analysis assistant. Answer the user's question using the provided Information. 

Guidelines:
1. Fact Retrieval: If the user asks a factual question, rely STRICTLY on the Information provided.
2. Analysis & Advice: If the user asks for feedback, summaries, or improvements (like reviewing a resume), evaluate the Information using your general knowledge, but keep your advice highly relevant to the provided text.
3. Citations: YOU MUST CITE YOUR SOURCES. Whenever you reference specific details from the text, append the exact source tag at the end of the sentence (e.g., [Source 1, p.4 - doc.pdf]).
4. Refusal: If the provided Information is completely blank or unrelated to the user's question, state: "I cannot answer this based on the provided documents."

Information:
{context}

Question:
{question}

Answer:
"""
        messages = [{"role": "system", "content": "You are a precise document analysis AI."}]
        for msg in history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        def generate():
            try:
                yield f"data: {json.dumps({'sources': sources})}\n\n"
                stream = rag.client.chat.completions.create(
                    model="openai/gpt-oss-120b:free",
                    messages=messages,
                    max_tokens=400,
                    stream=True,
                    temperature=0.2,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        yield f"data: {json.dumps({'token': content})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        logger.error("Error in /api/ask", exc_info=True)
        return jsonify({"error": str(e)}), 500
    
@app.route("/api/summarize", methods=["POST"])
def summarize_all():
    if rag.retriever is None:
        return jsonify({"error": "No PDFs uploaded yet. Please upload a PDF first."}), 400
    if reload_status["indexing"]:
        return jsonify({"error": "Still indexing. Please wait a moment."}), 503

    try:
        docs = rag.vectorstore.max_marginal_relevance_search(
            "introduction overview abstract summary purpose conclusions", 
            k=16,
            fetch_k=60
        )
        context, _ = build_context(docs)
        logger.info(f"Summarize triggered with {len(docs)} chunks")

        prompt = f"""Summarize the provided documents in a few short paragraphs. 
Make sure to cover the main purpose and key topics of ALL the different documents provided below. Be concise but comprehensive.

Context:
{context}

Summary:"""

        messages = [
            {"role": "system", "content": "You are a concise document assistant. Write short, clear summaries. Never use knowledge outside the provided context."},
            {"role": "user", "content": prompt},
        ]

        def generate():
            try:
                stream = rag.client.chat.completions.create(
                    model="openai/gpt-oss-120b:free",
                    messages=messages,
                    max_tokens=600,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    if content:
                        yield f"data: {json.dumps({'token': content})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error("Streaming error in /api/summarize", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        logger.error("Unhandled error in /api/summarize", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "ready": rag.retriever is not None
    })

load_rag()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)