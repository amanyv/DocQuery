import os, threading, logging, json
from flask import Blueprint, request, jsonify, send_from_directory, Response, stream_with_context
from werkzeug.utils import secure_filename
from supabase import create_client
from flask import redirect

import main as rag 

api = Blueprint('api', __name__)
logger = logging.getLogger("docquery")

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Docs")
os.makedirs(DOCS_DIR, exist_ok=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL else None

def set_indexing_status(is_indexing: bool, error_message: str = None):
    if not supabase: return
    try:
        supabase.table("system_status").upsert({
            "id": 1, "is_indexing": is_indexing, "error_message": error_message
        }).execute()
    except Exception as e:
        logger.error(f"Failed to update status in DB: {e}")

def get_indexing_status():
    if not supabase: return {"is_indexing": False, "error_message": None}
    try:
        response = supabase.table("system_status").select("*").eq("id", 1).execute()
        if response.data: return response.data[0]
    except Exception as e:
        logger.error(f"Failed to read status from DB: {e}")
    return {"is_indexing": False, "error_message": None}

def _background_worker(uploaded_files_data, user_id):
    set_indexing_status(True)
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
                            supabase_path, f, {"content-type": "application/pdf", "upsert": "true"}
                        )
                except Exception as e:
                    logger.error(f"Supabase upload failed: {e}")

        logger.info(f"Indexing new files for user {user_id}...")
        rag.add_documents(file_paths, user_id)
        set_indexing_status(False)
    except Exception as e:
        logger.exception("BACKGROUND WORKER FAILED")
        set_indexing_status(False, error_message=str(e))

def build_context(docs):
    parts = []
    sources = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        filename = doc.metadata.get("source_file", "document")
        label = f"[Source {i}, p.{page} - {filename}]"
        parts.append(f"{label}\n{doc.page_content.strip()}")
        sources.append({"index": i, "page": page, "file": filename})
    return "\n\n".join(parts), sources

def build_contextual_query(question, history):
    if not history: return question
    try:
        messages = [{"role": "system", "content": "Rewrite the user's follow-up question into a clear, standalone query based on the conversation history. Reply ONLY with the new query."}]
        for msg in history[-4:]: messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": f"Rewrite this: {question}"})
        resp = rag.client.chat.completions.create(model="openai/gpt-oss-120b:free", messages=messages, max_tokens=30, temperature=0.0)
        rewritten = resp.choices[0].message.content.strip()
        return rewritten if rewritten else question
    except Exception:
        return question

@api.route("/")
def index():
    return send_from_directory("static", "index.html")

@api.route("/api/status")
def status():
    db_status = get_indexing_status()
    indexing = db_status.get("is_indexing", False)
    error = db_status.get("error_message")
    ready = True if rag.vectorstore is not None else False
    msg = f"Error: {error}" if error else "Indexing PDF..." if indexing else "Pipeline ready." if ready else "Waiting..."
    return jsonify({"ready": ready, "indexing": indexing, "error": error, "message": msg})

@api.route("/api/upload", methods=["POST"])
def upload():
    user_id = request.headers.get("X-User-ID")
    if not user_id: return jsonify({"error": "Unauthorized"}), 401
    if get_indexing_status().get("is_indexing"): return jsonify({"error": "Already indexing."}), 429
    
    files = request.files.getlist("files")
    if not files: return jsonify({"error": "No files selected."}), 400

    user_dir = os.path.join(DOCS_DIR, user_id)
    os.makedirs(user_dir, exist_ok=True)
    uploaded_files_data = []

    for file in files:
        if file.filename.endswith(".pdf"):
            filename = secure_filename(file.filename)
            save_path = os.path.join(user_dir, filename)
            file.save(save_path)
            uploaded_files_data.append({"path": save_path, "filename": filename})

    if not uploaded_files_data: return jsonify({"error": "No valid PDFs."}), 400

    threading.Thread(target=_background_worker, args=(uploaded_files_data, user_id), daemon=True).start()
    return jsonify({"message": "Files uploaded. Indexing...", "files": [f['filename'] for f in uploaded_files_data]})

@api.route('/api/reset', methods=['POST'])
def reset_session():
    user_id = request.headers.get("X-User-ID")
    if not user_id: return jsonify({"error": "Unauthorized."}), 401
    try:
        set_indexing_status(False, None)
        if supabase:
            supabase.table(rag.TABLE_NAME).delete().eq("metadata->>user_id", user_id).execute()
            try:
                files_res = supabase.storage.from_("DocQuery").list(user_id)
                if files_res:
                    files_to_remove = [f"{user_id}/{f['name']}" for f in files_res if f['name'] != '.emptyFolderPlaceholder']
                    if files_to_remove: supabase.storage.from_("DocQuery").remove(files_to_remove)
            except Exception: pass
        
        user_dir = os.path.join(DOCS_DIR, user_id)
        if os.path.exists(user_dir):
            for filename in os.listdir(user_dir):
                file_path = os.path.join(user_dir, filename)
                if os.path.isfile(file_path): os.remove(file_path)
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@api.route("/api/files", methods=["GET"])
def list_files():
    user_id = request.headers.get("X-User-ID")
    if not user_id: return jsonify({"files": []})
    user_dir = os.path.join(DOCS_DIR, user_id)
    files = [f for f in os.listdir(user_dir) if f.endswith(".pdf")] if os.path.exists(user_dir) else []
    return jsonify({"files": files})

@api.route("/api/files/<filename>", methods=["GET"])
def get_file(filename):
    user_id = request.args.get("user") or request.headers.get("X-User-ID")
    if not user_id: 
        return jsonify({"error": "Unauthorized."}), 401
        
    try:
        res = supabase.storage.from_("DocQuery").create_signed_url(f"{user_id}/{filename}", 60)
        signed_url = res.get("signedURL")
        if not signed_url:
            return jsonify({"error": "File not found in storage."}), 404
        
        return redirect(signed_url)
    except Exception as e:
        logger.error(f"Error fetching PDF from Supabase: {e}")
        return jsonify({"error": "Failed to load PDF."}), 500

@api.route("/api/files/<filename>", methods=["DELETE"])
def delete_file(filename):
    user_id = request.headers.get("X-User-ID")
    if not user_id: return jsonify({"error": "Unauthorized."}), 401
    try:
        supabase.storage.from_("DocQuery").remove([f"{user_id}/{filename}"])
    except Exception: pass
    if rag.vectorstore: rag.delete_document(filename, user_id)
    path = os.path.join(DOCS_DIR, user_id, filename)
    if os.path.exists(path): os.remove(path)
    return jsonify({"message": f"Deleted {filename}"})

@api.route("/api/ask", methods=["POST"])
def ask():
    user_id = request.headers.get("X-User-ID")
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not user_id: return jsonify({"error": "Unauthorized."}), 401
    if not question: return jsonify({"error": "Question required."}), 400

    db_status = get_indexing_status()
    if db_status and db_status.get("error_message"):
        err = db_status.get("error_message")
        if "429" in err or "quota" in err.lower() or "ResourceExhausted" in err:
            return jsonify({"error": "⚠️ **API Quota Reached:** You have exceeded your free daily Gemini embedding requests. Please try again tomorrow or switch to local embeddings."}), 429
        return jsonify({"error": f"⚠️ **Indexing Failed:** {err}"}), 400
    
    if rag.vectorstore is None: return jsonify({"error": "System loading."}), 400

    try:
        history = data.get("history", [])
        query = build_contextual_query(question, history)
        
        query_embedding = rag.embeddings.embed_query(query)
        response = rag.supabase_client.rpc(rag.RPC_NAME, {
            "query_embedding": query_embedding, "match_count": 8, "filter": {"user_id": user_id}
        }).execute()

        from langchain_core.documents import Document
        docs = [Document(page_content=row["content"], metadata=row["metadata"]) for row in response.data]
        
        if not docs: return jsonify({"error": "No documents available."}), 400
        context, sources = build_context(docs)

        prompt = f"""
You are an intelligent document analysis assistant. Answer the user's question using the provided Information. 
Guidelines:
1. Fact Retrieval: Rely STRICTLY on the Information provided.
2. Citations: YOU MUST CITE YOUR SOURCES. Append the exact source tag at the end of the sentence (e.g., [Source 1, p.4 - doc.pdf]).
3. Refusal: If the provided Information is completely blank or unrelated, state: "I cannot answer this based on the provided documents."
4. Math Formatting: Use \\( and \\) for inline equations, and \\[ and \\] for block equations. NEVER use $ signs.

Information:
{context}

Question:
{question}

Answer:"""
        
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a precise document analysis AI. Answer the user's question using only the provided Information. "
                    "MATH FORMATTING RULES:\n"
                    "1. Use \\( and \\) for inline math (e.g., \\( x \\) or \\( Q, K, V \\)). NEVER use the $ symbol.\n"
                    "2. Use \\[ and \\] for standalone block equations. You MUST place empty blank lines before and after block equations. NEVER use $$.\n"
                    "3. Ensure pristine LaTeX syntax."
                    "4. If you see PDF extraction artifacts attached to emails or phone numbers (like the word 'envel~pe'), remove them and only output the clean email address."
                )
            }
        ]
        for msg in data.get("history", [])[-6:]: messages.append(msg)
        messages.append({"role": "user", "content": prompt})

        def generate():
            yield f"data: {json.dumps({'sources': sources})}\n\n"
            stream = rag.client.chat.completions.create(
                model="openai/gpt-oss-120b:free", messages=messages, max_tokens=400, stream=True, temperature=0.2
            )
            for chunk in stream:
                if getattr(chunk.choices[0].delta, "content", None):
                    yield f"data: {json.dumps({'token': chunk.choices[0].delta.content})}\n\n"
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route("/api/summarize", methods=["POST"])
def summarize_all():
    user_id = request.headers.get("X-User-ID")
    if not user_id: return jsonify({"error": "Unauthorized."}), 401

    db_status = get_indexing_status()
    if db_status and db_status.get("error_message"):
        err = db_status.get("error_message")
        if "429" in err or "quota" in err.lower() or "ResourceExhausted" in err:
            return jsonify({"error": "⚠️ **API Quota Reached:** You have exceeded your free daily Gemini embedding requests. Please try again tomorrow or switch to local embeddings."}), 429
        return jsonify({"error": f"⚠️ **Indexing Failed:** {err}"}), 400
    
    if rag.vectorstore is None: return jsonify({"error": "System loading."}), 400

    try:
        query_embedding = rag.embeddings.embed_query("introduction overview abstract summary purpose")
        response = rag.supabase_client.rpc(rag.RPC_NAME, {
            "query_embedding": query_embedding, "match_count": 16, "filter": {"user_id": user_id}
        }).execute()
        
        from langchain_core.documents import Document
        docs = [Document(page_content=row["content"], metadata=row["metadata"]) for row in response.data]
        if not docs: return jsonify({"error": "No PDFs uploaded."}), 400
        
        context, _ = build_context(docs)
        messages = [
            {
                "role": "system", 
                "content": (
                    "PRIMARY DIRECTIVE: You are a strict document summarizer. "
                    "ONLY summarize the text provided in the context. DO NOT invent tutorials or outside knowledge.\n\n"
                    "CLEAN TEXT RULES:\n"
                    "1. SCRUB ACADEMIC CITATIONS: Do NOT include original academic reference numbers, brackets, or bibliography citations (e.g., remove [25], [26], etc.) from your final text.\n"
                    "2. Do not include raw source tags like [Source 14, 0] in the summary output. Keep the paragraphs clean and readable.\n\n"
                    "3. SCRUB ARTIFACTS: If you see PDF extraction artifacts attached to emails or phone numbers (like the word 'envel~pe'), remove them completely.\n\n"
                    "SECONDARY DIRECTIVE (MATH FORMATTING):\n"
                    "1. Use \\( and \\) for inline math and sequences (e.g., \\( x \\)). NEVER use the $ symbol.\n"
                    "2. Use \\[ and \\] for standalone block equations. You MUST place empty blank lines before and after block equations. NEVER use $$.\n"
                    "3. Ensure pristine LaTeX syntax."
                )
            },
            {"role": "user", "content": f"Summarize the following document excerpts comprehensively:\n\n{context}"}
        ]

        def generate():
            stream = rag.client.chat.completions.create(
                model="openai/gpt-oss-120b:free", messages=messages, max_tokens=600, stream=True
            )
            for chunk in stream:
                if getattr(chunk.choices[0].delta, "content", None):
                    yield f"data: {json.dumps({'token': chunk.choices[0].delta.content})}\n\n"
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    except Exception as e:
        return jsonify({"error": str(e)}), 500