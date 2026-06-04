import os
import random
import time
import math
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

TOP_K = 8
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 25

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

IS_RENDER = os.getenv("RENDER") == "true"
TABLE_NAME = "documents_prod" if IS_RENDER else "documents_local"
RPC_NAME = "match_documents_prod" if IS_RENDER else "match_documents_local"

def init_rag():
    """Initializes the models and connects to Supabase pgvector using Gemini API embeddings."""
    global vectorstore, embeddings, supabase_client

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("🚨 CRITICAL: Missing SUPABASE_URL or SUPABASE_KEY in your .env file!")
    
    if not supabase_client:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

    if embeddings is None:
        env_label = "Render Production" if IS_RENDER else "Local Development"
        print(f"🚀 [{env_label}]: Initializing Cloud-Based Google Gemini Embeddings (3072 dims)...")
        
        key_pool = [
            os.getenv("GEMINI_API_KEY_1"),
            os.getenv("GEMINI_API_KEY_2"),
            os.getenv("GOOGLE_API_KEY")
        ]
        
        valid_keys = [k for k in key_pool if k]
        
        if not valid_keys:
            raise ValueError("🚨 CRITICAL: No Gemini API keys found in environment variables (Check GEMINI_API_KEY_1, GEMINI_API_KEY_2, or GOOGLE_API_KEY)!")
        
        selected_key = random.choice(valid_keys)
        print(f"🔑 Selected a Google API Key from a pool of {len(valid_keys)} configured keys.")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=selected_key
        )

    vectorstore = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name=TABLE_NAME,
        query_name=RPC_NAME
    )
    print(f"Connected to Supabase Vector Store Table: {TABLE_NAME}")

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
            success = False
            while not success:
                try:
                    vectorstore.add_documents(batch)
                    print(f"Uploaded batch {i//BATCH_SIZE + 1} of {math.ceil(len(chunks)/BATCH_SIZE)}...")
                    time.sleep(2)
                    success = True
                except Exception as e:
                    error_msg = str(e)

                    if "429" in error_msg or "Quota" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                        print("⚠️ Google API Free Tier limit reached! Pausing for 60 seconds before retrying...")
                        time.sleep(60)
                    else:
                        raise e
        
        print(f"Successfully added {len(chunks)} new chunks to Supabase for user {user_id}.")

def delete_document(filename: str, user_id: str):
    global supabase_client
    if supabase_client is None: return
    try:
        supabase_client.table(TABLE_NAME).delete().filter("metadata->>source_file", "eq", filename).filter("metadata->>user_id", "eq", user_id).execute()
        print(f"Successfully deleted vectors for {filename}")
    except Exception as e:
        print(f"Failed to delete vectors from Supabase: {e}")