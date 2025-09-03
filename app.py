import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pinecone_client import get_index
from prompt_templates import build_prompt


app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')


openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
INDEX_NAME = os.environ.get('PINECONE_INDEX', 'stem-mesa')
EMBED_MODEL = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')
CHAT_MODEL = os.environ.get('CHAT_MODEL', 'gpt-4o-mini')


idx = get_index(INDEX_NAME)


@app.get('/', response_class=HTMLResponse)
def home():
    return FileResponse('static/index.html')


@app.post('/chat')
async def chat(q: str = Form(...)):
    # 1) embed query
    q_emb = openai_client.embeddings.create(model=EMBED_MODEL, input=q).data[0].embedding
    # 2) query pinecone
    res = idx.query(vector=q_emb, top_k=4, include_metadata=True)
    contexts = [
        m['metadata'].get('text', '') if 'text' in m['metadata'] else m['metadata'].get('source', '')
        for m in res['matches']
    ]
    # Note: we didn't store 'text' in ingest to save space; you can store a short snippet in metadata if desired.
    # 3) build prompt and call chat
    prompt = build_prompt(contexts, q)
    completion = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=512
    )
    answer = completion.choices[0].message.content
    sources = [m['metadata'].get('source') for m in res['matches']]
    return {"answer": answer, "sources": sources}