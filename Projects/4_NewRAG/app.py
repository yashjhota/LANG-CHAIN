# app.py
import streamlit as st
from rag_cahin import ask

st.set_page_config(page_title="RAG", layout="centered")

st.title("🧠 Retrieval‑Augmented Generation ")
st.markdown(
    """
This uses **Groq** LLMs, **Chroma** vector store, and **LangChain** to answer questions based on the data you loaded.
"""
)

# Input box
question = st.text_input("Ask a question:", placeholder="e.g. Who is yash jain?")

if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        result = ask(question)

    answer = result["result"]
    sources = result["source_documents"]

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")
    for doc in sources:
        st.markdown(f"- {doc.metadata.get('source', 'unknown')}")