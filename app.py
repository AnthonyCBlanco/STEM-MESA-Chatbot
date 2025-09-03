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

# Load a local chat model (free)
generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

app = FastAPI()

def search(query, k=3):
    query_vec = embedder.encode([query])
    distances, indices = index.search(query_vec, k)
    results = [(texts[i], metadata[i]) for i in indices[0]]
    return results

@app.get("/chat")
def chat(q: str = Query(..., description="Your question")):
    results = search(q)
    context = "\n".join([r[0] for r in results])

    prompt = f"""
    You are a helpful assistant for the SBVC STEM-MESA Center.
    Answer the question based on the following documents:

    {context}

    Question: {q}
    """

    response = generator(prompt, max_new_tokens=200, do_sample=True)
    answer = response[0]["generated_text"]

    return {"answer": answer, "sources": [r[1]["source"] for r in results]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
