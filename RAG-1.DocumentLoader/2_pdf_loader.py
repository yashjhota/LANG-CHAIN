from langchain_community.document_loaders import PyPDFLoader



loader = PyPDFLoader('RAG-1.DocumentLoader\DEEP_LEARNING_PAPER.pdf')

docs = loader.load()

print(f"Number of documents: {len(docs)}")
print(f"Document metadata: {docs[0].metadata}")
# print(f"Document page content: {docs[0].page_content}")

