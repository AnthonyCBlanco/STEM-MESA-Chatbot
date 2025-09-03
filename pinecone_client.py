import os
import pinecone


PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')


def init_pinecone():
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)


def get_index(index_name: str, dimension: int = 1536):
    init_pinecone()
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension)
    return pinecone.Index(index_name)