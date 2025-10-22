import os
import sys
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from vector_store import VectorStore

MODEL_NAME = os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def read_text_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def ingest(path):
    text = read_text_file(path)
    chunks = chunk_text(text)
    print(f'Created {len(chunks)} chunks')

    model = SentenceTransformer(MODEL_NAME)
    embeddings = []
    metadatas = []
    for i, c in enumerate(tqdm(chunks, desc='Embedding')):
        emb = model.encode(c, convert_to_numpy=True)
        embeddings.append(emb)
        metadatas.append({'source': path, 'chunk_index': i, 'text': c})

    embeddings = np.vstack(embeddings).astype('float32')
    dim = embeddings.shape[1]
    vs = VectorStore(dim=dim)
    vs.build(embeddings, metadatas)
    print('Index built and saved')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ingest.py data.txt')
        sys.exit(1)
    ingest(sys.argv[1])
