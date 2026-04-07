import sys, os, time, shutil
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import threading
rag = None

def load_rag():
    global rag
    import main as rag_module
    rag = rag_module

threading.Thread(target=load_rag, daemon=True).start()

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Docs")

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/api/status")
def status():
    return jsonify({"ready": "True", "message": "RAG pipeline ready."})

@app.route("/api/upload", methods=["POST"])
def upload():
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
    
    try:
        rag.reload()
        return jsonify({"message": f"Uploaded and indexed: {', '.join(uploaded)}", "files": uploaded})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/files", methods=["GET"])
def list_files():
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    return jsonify({"files": files})

@app.route("/api/files/<filename>", methods=["DELETE"])
def delete_file(filename):
    path = os.path.join(DOCS_DIR, filename)
    if os.path.exists(path):
        os.remove(path)
        rag.reload()
        return jsonify({"message": f"Deleted {filename}"})
    return jsonify({"error": "File not found."}), 404

@app.route("/api/ask", methods=["POST"])
def ask():
    import traceback
    data = request.get_json()
    question = (data or {}).get("question", "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400
    
    if rag.retriever is None:
        return jsonify({"error": "No PDFs uploaded yet. Please upload a PDF first."}), 400

    MODELS = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-3-27b-it:free",
        "google/gemma-3-12b-it:free",
        "google/gemma-3-4b-it:free",
        "mistralai/mistral-7b-instruct:free",
        "qwen/qwen3-8b:free",
        "deepseek/deepseek-r1-zero:free",
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