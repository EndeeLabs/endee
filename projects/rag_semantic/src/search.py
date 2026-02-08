from sentence_transformers import SentenceTransformer
from vector_store import MockEndee

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Mock Endee
db = MockEndee()

# Load documents
with open("data/documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# Create embeddings and store them
embeddings = model.encode(documents)
for emb, doc in zip(embeddings, documents):
    db.add(emb, doc.strip())

print("✅ Documents loaded into vector store")

# Take user query
query = input("\nEnter your search query: ")

# Convert query to embedding
query_vector = model.encode([query])[0]

# Perform semantic search
results = db.search(query_vector, top_k=2)

print("\n🔍 Semantic Search Results:")
for i, res in enumerate(results, start=1):
    print(f"{i}. {res}")
