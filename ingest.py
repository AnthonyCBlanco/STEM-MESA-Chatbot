import os
import glob
import faiss
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

INDEX_DIR = "faiss_index"

# Load embedding model (free, local)
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_documents(source_dir="vault"):
    docs = []
    for filepath in glob.glob(os.path.join(source_dir, "**"), recursive=True):
        if filepath.endswith(".pdf"):
            reader = PdfReader(filepath)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            docs.append((filepath, text))
        elif filepath.endswith(".md") or filepath.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                docs.append((filepath, f.read()))
    return docs

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def ingest(source_dir="vault"):
    print("Loading documents...")
    docs = load_documents(source_dir)

    texts = []
    metadata = []

    for filepath, content in docs:
        chunks = chunk_text(content)
        texts.extend(chunks)
        metadata.extend([{"source": filepath}] * len(chunks))

    print("Creating embeddings...")
    embeddings = model.encode(texts)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump((texts, metadata), f)

    print(f"Ingested {len(texts)} chunks from {len(docs)} documents.")

if __name__ == "__main__":
    ingest()
