import sys, os, time, threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

rag = None
reload_status = {"running": False, "error": None}

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

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Docs")
os.makedirs(DOCS_DIR, exist_ok=True)

def _reload_in_background():
    """Run rag.reload() in a background thread and track status."""
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
 
    return jsonify({"ready": ready, "indexing": indexing, "error": error, "message": msg})

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
        msg = f"Uploaded {', '.join(uploaded)}. Indexing in background."
        indexing = True
    else:
        msg = f"Uploaded {', '.join(uploaded)}. Server still warming up — upload again in ~30s to index."
        indexing = False

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
        return jsonify({"error": "Server is still loading, please wait and try again."}), 503
    if rag.retriever is None:
        if reload_status["running"]:
            return jsonify({"error": "Still indexing your PDF. Please wait a moment and try again."}), 503
        return jsonify({"error": "No PDFs uploaded yet. Please upload a PDF first."}), 400

    MODELS = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-3-27b-it:free",
        "google/gemma-3-12b-it:free",
        "google/gemma-3-4b-it:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen3-8b:free",
        "microsoft/phi-4-reasoning:free",
    ]

    try:
        docs = rag.retriever.invoke(question)
        context = ""
        sources = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 0)
            context += f"[Source {i+1}: {source}, Page {page+1}]\n{doc.page_content}\n\n"
            sources.append({"file": os.path.basename(source), "page": page + 1})

        prompt = f"""You are a precise RAG assistant. Answer ONLY using the context below.
Format your response using markdown with these exact sections:

## Overview
2-3 sentence summary.

## Detailed Explanation
Thorough breakdown with bullet points.

## Key Points
- Key point 1
- Key point 2
- Key point 3

## Conclusion
1-2 sentence takeaway.

If the answer isn't in the context, say "I cannot find this in the provided documents."

Context:
{context}

Question: {question}

Answer:"""

        last_error = None
        for model in MODELS:
            try:
                print(f"Trying model: {model}")
                if "gemma" in model:
                    messages = [{"role": "user", "content": prompt}]
                else:
                    messages = [
                        {"role": "system", "content": "You are a precise assistant. Always use markdown formatting with ## headers and bullet points. Never use plain paragraphs."},
                        {"role": "user", "content": prompt}
                    ]
                response = rag.client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                answer = response.choices[0].message.content
                print(f"✓ Success with: {model}")
                return jsonify({"answer": answer, "sources": sources, "model": model})

            except Exception as e:
                print(f"✗ Failed {model}: {str(e)[:80]}")
                last_error = e
                if "429" in str(e) or "402" in str(e) or "404" in str(e) or "400" in str(e):
                    time.sleep(2)
                    continue
                else:
                    raise

        return jsonify({"error": "All models rate-limited. Try again in a minute."}), 503

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)