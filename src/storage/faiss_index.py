import faiss
import numpy as np


class FaissIndex:

    def __init__(self, vector_dim: int):
        self.vector_dim = vector_dim

        # cosine similarity index
        self.index = faiss.IndexFlatIP(vector_dim)

    def add(self, vector: np.ndarray):
        vector = np.array(vector).astype("float32").reshape(1, -1)

        # normalize for cosine similarity
        faiss.normalize_L2(vector)

        self.index.add(vector)

    def search(self, vector: np.ndarray, k: int = 3):

        if self.index.ntotal == 0:
            return [], []

        vector = np.array(vector).astype("float32").reshape(1, -1)

        faiss.normalize_L2(vector)

        scores, indices = self.index.search(vector, k)

        return scores[0], indices[0]

    def size(self):
        return self.index.ntotal

    def rebuild(self, embeddings):

        if not embeddings:
            return

        vectors = np.vstack(embeddings).astype("float32")

        # normalize all vectors
        faiss.normalize_L2(vectors)

        self.index = faiss.IndexFlatIP(self.vector_dim)
        self.index.add(vectors)


# Global shared FAISS instance
faiss_index = FaissIndex(vector_dim=768)