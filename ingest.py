import os
import glob
import faiss
import pickle
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Directory to save the index and metadata
INDEX_DIR = "data" 

# Load embedding model
try:
    print("Loading embedding model: all-MiniLM-L6-v2...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Failed to load embedding model: {e}")
    exit()


def load_documents(source_dir="vault"):
    """Loads text from PDF and TXT files."""
    docs = []
    print(f"Scanning for documents in '{source_dir}' directory...")
    filepaths = list(glob.glob(os.path.join(source_dir, "**/*.*"), recursive=True))
    
    for filepath in filepaths:
        if not os.path.isfile(filepath):
            continue
        
        text = ""
        try:
            if filepath.endswith(".pdf"):
                with open(filepath, "rb") as f:
                    reader = PdfReader(f)
                    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            elif filepath.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
            
            if text:
                docs.append({"filepath": filepath, "content": text})
                print(f"  - Loaded {os.path.basename(filepath)}")
        except Exception as e:
            print(f"Error processing file {filepath}: {e}")
    return docs

def chunk_documents(documents):
    """Splits documents into smaller, overlapping chunks."""
    all_chunks_with_metadata = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    
    for doc in documents:
        chunks = text_splitter.split_text(doc["content"])
        for chunk in chunks:
            all_chunks_with_metadata.append({
                "content": chunk,
                "source": os.path.basename(doc["filepath"]) # Store only the filename
            })
            
    return all_chunks_with_metadata

def ingest():
    """Main ingestion pipeline."""
    docs = load_documents()
    if not docs:
        print("\nNo documents found in 'vault' directory. Please add files to ingest.")
        return

    print("\nChunking documents...")
    chunks_with_metadata = chunk_documents(docs)
    texts = [item['content'] for item in chunks_with_metadata]
    
    print(f"\nCreating embeddings for {len(texts)} chunks...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)

    # Build and save FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(INDEX_DIR, "docs.faiss"))

    # Save metadata (chunks and their sources)
    with open(os.path.join(INDEX_DIR, "docs.pkl"), "wb") as f:
        pickle.dump(chunks_with_metadata, f)

    print(f"\n✅ Ingestion complete! Created {len(texts)} chunks from {len(docs)} documents.")
    print(f"Index and metadata saved to the '{INDEX_DIR}' directory.")

if __name__ == "__main__":
    ingest()
