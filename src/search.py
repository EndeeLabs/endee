import numpy as np
from embed import embed_text

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_search(query, vector_store, top_k=2):
    query_vec = embed_text([query])[0]
    scores = []

    for item in vector_store:
        score = cosine_similarity(query_vec, item["vector"])
        scores.append((score, item["text"]))

    scores.sort(reverse=True)
    return [text for _, text in scores[:top_k]]
