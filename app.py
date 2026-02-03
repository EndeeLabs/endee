import streamlit as st
from src.search import semantic_search
from src.rag import generate_answer
from src.embed import embed_text

documents = [
    "CMR Institute of Technology is located in Bengaluru.",
    "Attendance requirement is minimum 75 percent.",
    "Endee is a vector database used for semantic search."
]

vectors = embed_text(documents)
vector_store = []

for i, vec in enumerate(vectors):
    vector_store.append({
        "id": i,
        "text": documents[i],
        "vector": vec
    })

st.title("📄 AI Document Q&A using Endee")

question = st.text_input("Ask a question")

if question:
    context = semantic_search(question, vector_store)
    answer = generate_answer(context, question)
    st.write(answer)
