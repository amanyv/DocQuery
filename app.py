import sys, os, threading, logging, json
from dotenv import load_dotenv
load_dotenv()
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

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Docs")
os.makedirs(DOCS_DIR, exist_ok=True)

def set_indexing_status(is_indexing: bool, error_message: str = None):
    if not supabase:
        return
    try:
        supabase.table("system_status").upsert({
            "id": 1, 
            "is_indexing": is_indexing,
            "error_message": error_message
        }).execute()
    except Exception as e:
        logger.error(f"Failed to update status in DB: {e}")

def get_indexing_status():
    if not supabase:
        return {"is_indexing": False, "error_message": None}
    try:
        response = supabase.table("system_status").select("*").eq("id", 1).execute()
        if response.data:
            return response.data[0]
    except Exception as e:
        logger.error(f"Failed to read status from DB: {e}")
    return {"is_indexing": False, "error_message": None}

def load_rag():
    """Initialize RAG components once at startup."""
    try:
        logger.info("Initializing RAG module...")
        rag.init_rag()
        logger.info("RAG initialized successfully")
    except Exception:
        logger.error("RAG failed to initialize", exc_info=True)

def _background_worker(uploaded_files_data, user_id):
    """
    10. Move Supabase upload fully to background so UI unblocks instantly.
    """
    set_indexing_status(True)

    with reload_lock:
        try:

            file_paths = []
            for file_data in uploaded_files_data:
                path = file_data["path"]
                filename = file_data["filename"]
                file_paths.append(path)

                if supabase:
                    try:
                        supabase_path = f"{user_id}/{filename}"
                        with open(path, "rb") as f:
                            supabase.storage.from_("DocQuery").upload(
                                filename, f, {"content-type": "application/pdf", "upsert": "true"}
                            )
                        logger.info(f"Uploaded to SupabaseStorage: {supabase_path}")
                    except Exception as e:
                        logger.error(f"Supabase upload failed: {e}")

            logger.info("Indexing new files into Vector DBfor user {user_id}...")
            rag.add_documents(file_paths, user_id)
            
            logger.info("Background indexing complete.")
            set_indexing_status(False)

        except Exception as e:
            logger.exception("BACKGROUND WORKER FAILED")
            set_indexing_status(False, error_message=str(e))

OVERVIEW_KEYWORDS = {"about", "overview", "summary", "summarize", "describe", "explain this", "tell me about", "what does it cover"}

def get_docs_for_question(query: str, user_id: str):
    logger.info("RETRIEVAL | query=%s | user=%s", query, user_id)
    q_lower = query.lower()
    is_overview = any(kw in q_lower for kw in OVERVIEW_KEYWORDS)

    search_query = "introduction overview summary purpose topics covered" if is_overview else query

    query_embedding = rag.embeddings.embed_query(search_query)

    response = rag.supabase_client.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": 8,
            "filter": {"user_id": user_id}
        }
    ).execute()

    from langchain_core.documents import Document
    initial_docs = [
        Document(page_content=row["content"], metadata=row["metadata"]) 
        for row in response.data
    ]

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
    db_status = get_indexing_status()
    indexing = db_status.get("is_indexing", False)
    error = db_status.get("error_message")
    
    ready = True if rag.vectorstore is not None else False
    msg = f"Indexing failed: {error}" if error else "Indexing PDF..." if indexing else "Pipeline ready." if ready else "Waiting..."
    
    return jsonify({"ready": ready, "indexing": indexing, "error": error, "message": msg})

@app.route("/api/upload", methods=["POST"])
def upload():
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return jsonify({"error": "Unauthorized: No User ID provided."}), 401
        
    db_status = get_indexing_status()
    if db_status.get("is_indexing"):
        return jsonify({"error": "Already indexing, please wait."}), 429
        
    if "files" not in request.files:
        return jsonify({"error": "No files provided."}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files selected."}), 400

    user_dir = os.path.join(DOCS_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)

    uploaded = []
    uploaded_files_data = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            continue
        
        filename = secure_filename(file.filename)
        save_path = os.path.join(user_dir, filename)

        if os.path.exists(save_path) and rag.vectorstore:
            rag.delete_document(save_path)
            os.remove(save_path)

        file.save(save_path)
        uploaded_files_data.append({"path": save_path, "filename": filename})
        uploaded.append(filename)

    if not uploaded:
        return jsonify({"error": "No valid PDF files found."}), 400

    threading.Thread(target=_background_worker, args=(uploaded_files_data, user_id), daemon=True).start()
    return jsonify({"message": "Files uploaded. Indexing in background.", "files": uploaded, "indexing": True})

@app.route('/api/reset', methods=['POST'])
def reset_session():
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return jsonify({"error": "Unauthorized."}), 401

    try:
        set_indexing_status(False, None)
        
        if supabase:
            supabase.table("documents").delete().eq("metadata->>user_id", user_id).execute()
            
            try:
                files_res = supabase.storage.from_("DocQuery").list(user_id)
                if files_res:
                    files_to_remove = [f"{user_id}/{f['name']}" for f in files_res if f['name'] != '.emptyFolderPlaceholder']
                    if files_to_remove:
                        supabase.storage.from_("DocQuery").remove(files_to_remove)
            except Exception as e:
                logger.error(f"Failed to clear Supabase storage on reset: {e}")

        user_dir = os.path.join(DOCS_DIR, user_id)
        if os.path.exists(user_dir):
            for filename in os.listdir(user_dir):
                file_path = os.path.join(user_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        return jsonify({"status": "success", "message": "Environment reset successfully"}), 200
        
    except Exception as e:
        logger.error(f"Session reset failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/files", methods=["GET"])
def list_files():
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return jsonify({"files": []})
        
    user_dir = os.path.join(DOCS_DIR, user_id)
    if not os.path.exists(user_dir):
        return jsonify({"files": []})
    
    files = [f for f in os.listdir(user_dir) if f.endswith(".pdf")]
    return jsonify({"files": files})

@app.route("/api/files/<filename>", methods=["DELETE"])
def delete_file(filename):
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return jsonify({"error": "Unauthorized."}), 401
    
    try:
        supabase_path = f"{user_id}/{filename}"
        supabase.storage.from_("DocQuery").remove([supabase_path])
    except Exception as e:
        logger.error(f"Failed to remove from storage: {e}")

    if rag.vectorstore is not None:
        rag.delete_document(filename, user_id)
        
    path = os.path.join(DOCS_DIR, user_id, filename)
    if os.path.exists(path):
        os.remove(path)

    return jsonify({"message": f"Deleted {filename}"})

@app.route("/api/ask", methods=["POST"])
def ask():
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return jsonify({"error": "Unauthorized."}), 401
    
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question is required."}), 400
    
    if rag.vectorstore is None:
        return jsonify({"error": "No documents available. Upload a PDF."}), 400

    try:
        history = data.get("history", [])
        
        query = build_contextual_query(question, history)

        docs = get_docs_for_question(query, user_id)

        if not docs:
            return jsonify({"error": "No documents available. Upload a PDF."}), 400
        
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
    user_id = request.headers.get("X-User-ID")
    if not user_id:
        return jsonify({"error": "Unauthorized."}), 401

    if rag.vectorstore is None:
        return jsonify({"error": "System loading."}), 400

    try:
        query_embedding = rag.embeddings.embed_query("introduction overview abstract summary purpose conclusions")
        response = rag.supabase_client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_count": 16,
                "filter": {"user_id": user_id}
            }
        ).execute()
        
        from langchain_core.documents import Document
        docs = [
            Document(page_content=row["content"], metadata=row["metadata"]) 
            for row in response.data
        ]
        
        if not docs:
            return jsonify({"error": "No PDFs uploaded yet. Please upload a PDF first."}), 400
        
        context, _ = build_context(docs)

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
    })

load_rag()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, threaded=True)