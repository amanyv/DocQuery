import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Docs")
DB_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

TOP_K = 8
FETCH_K = 20
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 100

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
reranker    = None

def init_rag():
    """Initializes the models and loads existing DB on startup."""
    global vectorstore, retriever, embeddings, reranker
    
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)

    if embeddings is None:
        print("Connecting to Google Gemini Embedding API...")
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not set in environment variables")
            
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_key
        )

    # if reranker is None:
    #     print("Loading reranker model (CrossEncoder)...")
    #     reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
        collection_name="docquery"
    )

    existing_ids = vectorstore.get()["ids"]
    if len(existing_ids) > 0:
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K, "fetch_k": FETCH_K}
        )
        print("Existing vector DB loaded.")

def add_documents(file_paths):
    """Processes ONLY the specific new files uploaded by the user."""
    global vectorstore, retriever
    
    if not file_paths:
        return

    print(f"Incremental Indexing: Processing {len(file_paths)} new file(s)...")
    new_docs = []
    
    for path in file_paths:
        abs_path = os.path.abspath(path)
        loader = PyPDFLoader(abs_path)
        new_docs.extend(loader.load())

    if new_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(new_docs)

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            vectorstore.add_documents(batch)
        
        print(f"Successfully added {len(chunks)} new chunks to the database.")

    retriever = vectorstore.as_retriever(
        search_type="mmr", 
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K}
    )

def delete_document(source_path: str):
    """Batch deletes chunks to bypass SQLite limits."""
    global vectorstore, retriever

    if vectorstore is None:
        return

    source_path = os.path.abspath(source_path)
    results = vectorstore.get()
    ids_to_delete = []

    for i, meta in enumerate(results["metadatas"]):
        stored = os.path.abspath(meta.get("source", ""))
        if stored == source_path:
            ids_to_delete.append(results["ids"][i])

    if not ids_to_delete:
        print(f"No chunks found for {source_path}")
        return

    for i in range(0, len(ids_to_delete), 500):
        batch_ids = ids_to_delete[i:i+500]
        vectorstore.delete(ids=batch_ids)
        
    print(f"Deleted {len(ids_to_delete)} chunks for {source_path}")

    if len(vectorstore.get()["ids"]) == 0:
        retriever = None
    else:
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K, "fetch_k": FETCH_K})

if __name__ == "__main__":
    init_rag()