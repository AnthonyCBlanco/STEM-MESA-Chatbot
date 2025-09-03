
import os
from typing import List
from pypdf import PdfReader
import markdown


# simple text extraction from PDFs and markdown


def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(pages)




def extract_text_from_md(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()




def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(' '.join(chunk))
        i += chunk_size - overlap
    return chunks