# STEM-MESA RAG Chatbot (Free / Local Version)

This chatbot lets students ask questions about STEM-MESA services at SBVC.

## How it works
- Stores flyers and docs in `vault/`
- Ingests them into FAISS with local embeddings (sentence-transformers)
- Uses Hugging Face LLM (Mistral) for chat responses
- Exposes an API with FastAPI

## Usage
1. Put PDFs/Markdown in `vault/`
2. Ingest files:
   ```bash
   python ingest.py
   ```
3. Start chatbot:
   ```bash
   uvicorn app:app --reload
   ```
4. Ask questions:
   - Open: http://127.0.0.1:8000/chat?q=What are the tutoring hours?
