import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class MockEndee:
    """
    A lightweight mock of Endee vector database.
    Stores embeddings and performs cosine similarity search.
    """

    def __init__(self):
        self.vectors = []
        self.texts = []

    def add(self, vector, text):
        self.vectors.append(vector)
        self.texts.append(text)

    def search(self, query_vector, top_k=2):
        if len(self.vectors) == 0:
            return []

        similarities = cosine_similarity(
            [query_vector],
            self.vectors
        )[0]

        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.texts[i] for i in top_indices]
