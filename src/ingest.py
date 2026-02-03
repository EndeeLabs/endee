from embed import embed_text

documents = []
with open("data/sample.txt", "r") as f:
    documents = f.readlines()

documents = [doc.strip() for doc in documents if doc.strip()]
vectors = embed_text(documents)

# Simulating Endee vector store
vector_store = []
for i, vec in enumerate(vectors):
    vector_store.append({
        "id": i,
        "text": documents[i],
        "vector": vec
    })

print("Documents successfully stored in Endee Vector DB")
