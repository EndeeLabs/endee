from sentence_transformers import SentenceTransformer
from vector_store import MockEndee

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Mock Endee
db = MockEndee()

# Load documents
with open("data/documents.txt", "r", encoding="utf-8") as f:
    documents = f.readlines()

# Store document embeddings
embeddings = model.encode(documents)
for emb, doc in zip(embeddings, documents):
    db.add(emb, doc.strip())

print("✅ Documents indexed for RAG")

# Ask user question
query = input("\nAsk a question: ")

# Embed query
query_vector = model.encode([query])[0]

# Retrieve relevant documents
retrieved_docs = db.search(query_vector, top_k=2)
context = " ".join(retrieved_docs)

# ---- MOCK LLM RESPONSE ----
print("\n🤖 Answer:")

if not context.strip():
    print("No relevant information found in the documents.")
else:
    print(
        "Based on the retrieved context, a vector database stores numerical "
        "embeddings of data and enables semantic similarity search, allowing "
        "applications like semantic search and Retrieval Augmented Generation "
        "(RAG) systems to retrieve relevant information efficiently."
    )
