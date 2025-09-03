import pickle
import faiss
import uvicorn
from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
from transformers import pipeline

INDEX_DIR = "faiss_index"

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index + metadata
index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
with open(f"{INDEX_DIR}/metadata.pkl", "rb") as f:
    texts, metadata = pickle.load(f)

# Load a free instruction-tuned Hugging Face model for better answers
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=200)

app = FastAPI()

def search(query, k=2):
    """
    Return top-k relevant document chunks from FAISS.
    """
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec, k)
    results = [(texts[i], metadata[i]) for i in indices[0]]
    return results

@app.get("/chat")
def chat(q: str = Query(..., description="Your question")):
    """
    Chat endpoint: returns a concise answer and source documents.
    """
    results = search(q)
    context = "\n".join([r[0] for r in results])

    # Short instruction-tuned prompt
    prompt = f"""
Answer the following question based on these documents. Be concise and clear.

Documents:
{context}

Question: {q}
Answer:
"""

    response = generator(prompt, max_new_tokens=200)
    full_text = response[0]["generated_text"]

    # Remove the prompt portion from the output if repeated
    answer = full_text.replace(prompt, "").strip()

    return {"answer": answer, "sources": [r[1]["source"] for r in results]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
