import os
import json
import faiss
import numpy as np

class VectorStore:
    def __init__(self, index_path='index.faiss', meta_path='metadata.json', dim=384):
        self.index_path = index_path
        self.meta_path = meta_path
        self.dim = dim
        self.index = None
        self.metadata = []

    def _ensure_index(self):
        if self.index is None:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            else:
                self.index = faiss.IndexFlatIP(self.dim)

    def save(self):
        self._ensure_index()
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self):
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)

    def build(self, embeddings: np.ndarray, metadatas: list):
        """Build a new index from embeddings and metadatas. embeddings shape (N, dim)"""
        assert embeddings.ndim == 2 and embeddings.shape[1] == self.dim
        # normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embeddings)
        self.metadata = metadatas
        self.save()

    def search(self, query_emb: np.ndarray, k=5):
        """Search top-k nearest. query_emb shape (dim,) or (1,dim)"""
        self._ensure_index()
        if self.index is None or self.index.ntotal == 0:
            return []
        if query_emb.ndim == 1:
            q = query_emb.reshape(1, -1).astype('float32')
        else:
            q = query_emb.astype('float32')
        faiss.normalize_L2(q)
        D, I = self.index.search(q, k)
        results = []
        for dist_row, idx_row in zip(D, I):
            row = []
            for dist, idx in zip(dist_row, idx_row):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                md = self.metadata[idx]
                row.append({'score': float(dist), 'metadata': md})
            results.append(row)
        return results
