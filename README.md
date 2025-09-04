STEM-MESA RAG Chatbot (Upgraded)
This chatbot provides conversational answers to student questions about STEM-MESA services at SBVC using a local, lightweight AI model.

How it works
Stores program flyers, schedules, and documents in the vault/ directory.

Uses langchain to intelligently split documents into context-aware chunks.

Ingests document chunks into a FAISS vector index using sentence-transformers for semantic search.

Uses the lightweight and powerful microsoft/phi-3-mini-4k-instruct generative model to provide conversational answers based on retrieved context.

Exposes a simple API with FastAPI and includes a web interface.

Usage
Install Dependencies:
Make sure you have all the required packages installed from requirements.txt.

pip install -r requirements.txt

Add Documents:
Place your PDF, Markdown, or TXT files into the vault/ directory.

Ingest Documents:
Run the ingestion script to process your documents and build the search index. You only need to do this when you add or change documents.

python ingest.py

Start the Chatbot Server:

uvicorn app:app --reload

Chat with the Bot:
Open your web browser and navigate to https://www.g