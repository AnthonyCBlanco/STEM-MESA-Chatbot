import os
import argparse
import uuid
from openai import OpenAI
from tqdm import tqdm
from utils import extract_text_from_md, extract_text_from_pdf, chunk_text
from pinecone_client import get_index

# lightweight ingestion script

def embed_texts(texts, openai_client, model):
    # batch embeddings
    resp = openai_client.embeddings.create(model=model, input=texts)
    return [r.embedding for r in resp.data]

def main(source, index_name, model, chunk_size, overlap):
    openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    idx = get_index(index_name)
    files = []
    for root, _, names in os.walk(source):
        for n in names:
            if n.lower().endswith('.md') or n.lower().endswith('.pdf'):
                files.append(os.path.join(root, n))

    for fpath in tqdm(files):
        if fpath.endswith('.md'):
            text = extract_text_from_md(fpath)
        else:
            text = extract_text_from_pdf(fpath)
        chunks = chunk_text(text, chunk_size, overlap)
        embeddings = embed_texts(chunks, openai_client, model)
        # prepare upserts
        vectors = []
        for i, emb in enumerate(embeddings):
            meta = {"source": fpath, "chunk_index": i}
            vectors.append((str(uuid.uuid4()), emb, meta))
        # upsert in batches
        for i in range(0, len(vectors), 100):
            batch = vectors[i:i+100]
            idx.upsert(vectors=batch)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='./vault')
    p.add_argument('--index-name', default=os.environ.get('PINECONE_INDEX', 'stem-mesa'))
    p.add_argument('--model', default=os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small'))
    p.add_argument('--chunk-size', type=int, default=int(os.environ.get('CHUNK_SIZE', 800)))
    p.add_argument('--overlap', type=int, default=int(os.environ.get('CHUNK_OVERLAP', 100)))
    args = p.parse_args()
    main(args.source, args.index_name, args.model, args.chunk_size, args.overlap)