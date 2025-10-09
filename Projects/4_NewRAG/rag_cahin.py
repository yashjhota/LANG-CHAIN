import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# 1️⃣ Embeddings
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# 2️⃣ Load Chroma DB
CHROMA_PATH = "chroma_db"
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

# 3️⃣ LLM (Groq)
llm = ChatGroq(
    model_name="openai/gpt-oss-20b",
    temperature=0.2
)

# 4️⃣ Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an AI assistant. Use the following context to answer the question. 
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
""",
)

# 5️⃣ RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt},
)

# 6️⃣ Ask Function
def ask(question: str) -> dict:
    """Return answer + sources."""
    return qa_chain({"query": question})
