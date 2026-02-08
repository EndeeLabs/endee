from sentence_transformers import SentenceTransformer
from vector_store import MockEndee

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Mock Endee vector database
db = MockEndee()

# Read documents
with open("data/documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# Generate embeddings
embeddings = model.encode(documents)

# Store embeddings in vector database
for emb, doc in zip(embeddings, documents):
    db.add(emb, doc.strip())

print("✅ Documents successfully ingested into Mock Endee")
