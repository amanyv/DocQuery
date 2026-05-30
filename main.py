import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

TOP_K = 8
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
embeddings  = None
supabase_client: Client = None

def init_rag():
    """Initializes the models and connects to Supabase pgvector."""
    global vectorstore, embeddings, supabase_client

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("🚨 CRITICAL: Missing SUPABASE_URL or SUPABASE_KEY in your .env file!")
    
    if not supabase_client:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

    if embeddings is None:
        print("Initializing Local HuggingFace Embeddings...")
        gemini_key = os.getenv("GEMINI_API_KEY")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
        )

    vectorstore = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents"
    )
    print("Connected to Supabase Vector Store.")

def add_documents(file_paths, user_id):
    """Processes new files, tags them, and uploads vectors to Supabase."""
    global vectorstore
    
    if not file_paths: return

    print(f"Incremental Indexing: Processing {len(file_paths)} file(s) for user {user_id}...")
    new_docs = []
    
    for path in file_paths:
        abs_path = os.path.abspath(path)
        loader = PyPDFLoader(abs_path)
        docs = loader.load()
        
        for doc in docs:
            doc.metadata["user_id"] = user_id
            doc.metadata["source_file"] = os.path.basename(path)
        new_docs.extend(docs)

    if new_docs:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(new_docs)

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i+BATCH_SIZE]
            vectorstore.add_documents(batch)
        
        print(f"Successfully added {len(chunks)} new chunks to Supabase for user {user_id}.")

def delete_document(filename: str, user_id: str):
    """Directly deletes a document's chunks from Supabase using SQL filters."""
    global supabase_client
    if supabase_client is None: return
    
    try:
        supabase_client.table("documents").delete().eq("metadata->>source_file", filename).eq("metadata->>user_id", user_id).execute()
        print(f"Successfully deleted vectors for {filename}")
    except Exception as e:
        print(f"Failed to delete vectors from Supabase: {e}")