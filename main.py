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
            model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
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

    print("Rebuilding vector store...")
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    existing_files = set()
    if vectorstore._collection.count() > 0:
        metadatas = vectorstore.get()["metadatas"]
        existing_files = set(m.get("source") for m in metadatas if m.get("source"))

    new_docs = [
        d for d in documents
        if d.metadata.get("source") not in existing_files
    ]

    print(f"New documents to add: {len(new_docs)}")

    if new_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=120
        )
        chunks = splitter.split_documents(new_docs)
        vectorstore.add_documents(chunks)
        vectorstore.persist()
        print(f"Added {len(chunks)} chunks")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    print("  Vector store ready")

def delete_document(source_path: str):
    """Remove all chunks belonging to a specific source file from the vector store."""
    global vectorstore, retriever

    if vectorstore is None:
        return

    results = vectorstore.get(where={"source": source_path})
    ids_to_delete = results["ids"]

    if not ids_to_delete:
        print(f"No chunks found for {source_path}")
        return

    vectorstore.delete(ids=ids_to_delete)
    print(f"Deleted {len(ids_to_delete)} chunks for {source_path}")

    if vectorstore._collection.count() == 0:
        vectorstore = None
        retriever = None
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

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