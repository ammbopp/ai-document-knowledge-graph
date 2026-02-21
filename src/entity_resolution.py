# entity_resolution.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class EntityResolver:
    def __init__(self, threshold=0.85):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.entities = []
        self.embeddings = []
        self.threshold = threshold

    def resolve(self, entity_name: str) -> str:
        emb = self.model.encode([entity_name])

        if not self.entities:
            self.entities.append(entity_name)
            self.embeddings.append(emb)
            return entity_name

        sims = cosine_similarity(emb, np.vstack(self.embeddings))[0]
        best_idx = np.argmax(sims)

        if sims[best_idx] >= self.threshold:
            return self.entities[best_idx]

        self.entities.append(entity_name)
        self.embeddings.append(emb)
        return entity_name
