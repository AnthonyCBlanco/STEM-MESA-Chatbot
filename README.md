
LLM API

<img src="https://www.google.com/search?q=https://img.shields.io/badge/Groq-Llama_3.1--8B-F05A28%3Fstyle%3Dflat-square" alt="Groq Badge"/>

Unmatched speed (~800 tokens/s) and extremely low cost (~$0.05 / 1M tokens).

Backend

<img src="https://www.google.com/search?q=https://img.shields.io/badge/FastAPI-009688%3Fstyle%3Dflat-square%26logo%3Dfastapi%26logoColor%3Dwhite" alt="FastAPI Badge"/>

High-performance asynchronous Python framework for building the API.

Embeddings

<img src="https://www.google.com/search?q=https://img.shields.io/badge/Sentence--Transformers-3B82F6%3Fstyle%3Dflat-square" alt="Sentence-Transformers Badge"/>

Free, high-quality, local model (all-MiniLM-L6-v2) to convert text to vectors without API cost.

Vector Store

<img src="https://www.google.com/search?q=https://img.shields.io/badge/FAISS-4D90FE%3Fstyle%3Dflat-square" alt="FAISS Badge"/>

Fast, simple, and efficient local vector database for semantic search.

Frontend

<img src="https://www.google.com/search?q=https://img.shields.io/badge/HTML5-E34F26%3Fstyle%3Dflat-square%26logo%3Dhtml5%26logoColor%3Dwhite" alt="HTML5 Badge"/>

A clean, single-page web interface for interacting with the chatbot.

Styling

<img src="https://img.shields.io/badge/Tailwind_CSS-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white" alt="Tailwind CSS Badge"/>

A utility-first CSS framework for rapid UI development.

⚙️ Getting Started
Follow these steps to set up and run the chatbot on your local machine.

1. Prerequisites
Python 3.8+

A free Groq API Key

2. Installation
Clone the repository to your local machine:

git clone [https://github.com/your-username/sbvc-mesa-chatbot.git](https://github.com/your-username/sbvc-mesa-chatbot.git)
cd sbvc-mesa-chatbot

Create a virtual environment and install the required packages:

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

3. Configuration
Create a .env file in the root directory of the project and add your Groq API key:

GROQ_API_KEY="gsk_YourSecretGroqApiKeyGoesHere"

4. Add Your Documents
Place your knowledge base files (.pdf, .txt) into the vault/ directory. If this folder doesn't exist, create it.

mkdir vault
# Now add your files into the vault/ folder

Examples: tutoring_schedule.pdf, workshop_info.txt, etc.

5. Ingest the Data
Run the ingestion script to process your documents. This will create a data/ directory containing the vector index and metadata.

You must run this script every time you add, change, or remove documents in the vault/ directory.

python ingest.py

6. Run the Application
Start the FastAPI server using Uvicorn:

uvicorn app:app --reload

🖥️ Usage
Once the server is running, open your web browser and navigate to:

http://127.0.0.1:8000

You can now start asking questions in the chat interface!

📂 Project Structure
.
├── app.py                # The main FastAPI application logic
├── ingest.py             # Script to process and index documents
├── index.html            # The single-page frontend
├── requirements.txt      # Python dependencies
├── README.md             # You are here!
├── .env                  # Stores the API key (create this yourself)
├── vault/                # Directory for your source documents (PDFs, TXT)
└── data/                 # Stores the FAISS index and metadata (created by ingest.py)
