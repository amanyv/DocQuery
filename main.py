import os, shutil
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

    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    os.makedirs(DB_DIR, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    print("  Vector store ready")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

print("RAG will load on first upload...")

if __name__ == "__main__":
    reload()
    print("\nRAG ready! Type 'quit' to exit.\n")
    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        if retriever is None:
            print("No documents uploaded.")
            continue
        docs = retriever.invoke(question)
        for doc in docs:
            print(doc.page_content[:300])
            print("---")