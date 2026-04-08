import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Docs")
DB_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

vectorstore = None
retriever   = None
embeddings  = None

def reload():
    global vectorstore, retriever, embeddings

    if embeddings is None:
        print("Loading embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        print("Embedding model ready")

    pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print("  No PDFs found, skipping vector store build.")
        vectorstore = None
        retriever = None
        return
    
    print("Loading pdfs...")
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    documents = loader.load()
    print(f"  Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    print("Rebuilding vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("  Vector store ready")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

print("RAG will load on first upload...")

def ask(question):
    if retriever is None:
        return "No documents uploaded."
    
    docs = retriever.invoke(question)
    context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)
        context += f"[Source {i+1}: {source}, Page {page+1}]\n{doc.page_content}\n\n"

    prompt = f"""You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {question}"""

    response = client.chat.completions.create(
        model="meta-llama/llama-3.3-70b-instruct:free",
        messages=[
            {"role": "system", "content": "You are a precise assistant. Always use markdown with ## headers and bullet points."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content
    return answer

if __name__ == "__main__":
    print("\nRAG ready! Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        ask(question)