import os
import pickle
import faiss
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from sentence_transformers import SentenceTransformer
from groq import Groq, AsyncGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# --- Load Models and Index on startup ---
INDEX_DIR = "data"
embedding_model = None
index = None
chunks_with_metadata = None
groq_client = None

@app.on_event("startup")
async def startup_event():
    global embedding_model, index, chunks_with_metadata, groq_client

    print("--- Initializing Chatbot ---")
    
    # 1. Initialize Groq Client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("🔴 FATAL: GROQ_API_KEY environment variable not set.")
    else:
        groq_client = AsyncGroq(api_key=api_key)
        print("✅ Groq client initialized.")
        
    # 2. Load the embedding model
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model loaded.")
    except Exception as e:
        print(f"🔴 Error loading SentenceTransformer model: {e}")

    # 3. Load the FAISS index and metadata
    try:
        index_path = os.path.join(INDEX_DIR, "docs.faiss")
        metadata_path = os.path.join(INDEX_DIR, "docs.pkl")
        
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            chunks_with_metadata = pickle.load(f)
        print("✅ FAISS index and metadata loaded.")
    except Exception as e:
        print(f"🔴 FAISS index/metadata not found. Please run ingest.py. Error: {e}")
    
    print("--- Initialization Complete ---")

def search_index(query: str, k: int = 5):
    """Performs semantic search."""
    if embedding_model is None or index is None:
        print("Search components not ready.")
        return [], []

    query_embedding = embedding_model.encode(query, convert_to_tensor=False)
    query_embedding = np.array([query_embedding]).astype("float32")
    
    distances, indices = index.search(query_embedding, k)
    
    results = [chunks_with_metadata[i] for i in indices[0] if i < len(chunks_with_metadata)]
    
    # Get unique sources
    sources = sorted(list(set(item['source'] for item in results)))
    return results, sources

async def run_chat(question: str):
    """Main RAG pipeline."""
    if not groq_client or not index:
        return JSONResponse(
            status_code=503,
            content={"answer": "Chatbot is not fully initialized. Please check server logs.", "sources": []}
        )

    # 1. Retrieve relevant context
    context_chunks, sources = search_index(question, k=5)
    if not context_chunks:
        return JSONResponse({
            "answer": "I couldn't find any relevant information in my documents to answer that. Please contact the STEM-MESA front desk for help.",
            "sources": []
        })

    combined_context = "\n\n".join([item['content'] for item in context_chunks])

    # 2. Build the prompt
    system_prompt = """
    You are the SBVC STEM-MESA Center assistant.
    - Answer ONLY from the provided context.
    - If the answer is not in the context, say “I don’t know, please contact the STEM-MESA front desk.”
    - Be friendly, concise, and helpful.
    - Never reference the context directly in your answer.
    - Never ask for personal information.
    - Never Give answers to Chemisty, Math, Physics, Biology directyly, walk students through the steps to solve the problem. 
    """
    
    user_prompt = f"""
    Context:
    ---
    {combined_context}
    ---
    Question: {question}
    """
    
    # 3. Call the Groq API
    try:
        chat_completion = await groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=512,
        )
        answer = chat_completion.choices[0].message.content

    except Exception as e:
        print(f"🔴 Groq API error: {e}")
        return JSONResponse(
            status_code=500,
            content={"answer": "Sorry, I encountered an error while trying to generate a response.", "sources": []}
        )

    return JSONResponse({"answer": answer, "sources": sources})

@app.get("/")
async def serve_frontend():
    """Serves the main HTML file."""
    return FileResponse("index.html")

@app.post("/chat")
async def chat_endpoint(request: Request):
    """Main chat endpoint."""
    body = await request.json()
    question = body.get("question")
    if not question:
        return JSONResponse({"error": "Question not provided"}, status_code=400)
    
    return await run_chat(question)