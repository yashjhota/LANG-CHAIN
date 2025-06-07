from langchain_huggingface import HuggingFaceEmbeddings

embd = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
documents = [
    "Hello world",
    "Goodbye world",
    "Hello again",
    "Goodbye again",
]

# result = embd.embed_query(text)
result = embd.embed_documents(documents)
print(str(result))

