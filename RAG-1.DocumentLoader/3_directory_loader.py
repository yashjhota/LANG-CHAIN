# change the path and run please okay 

from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
#
loader= DirectoryLoader(
    path="C:/Users/jhota/OneDrive/Apps/Desktop/ALGO-BOOKS/algorithm-books",
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)

# # C:/Users/jhota/Downloads/DL Map
# docs = loader.load() 

# Lazy loading of documents

docs = loader.lazy_load()
for doc in docs:
    print(doc.metadata)

# print(f"Number of documents: {len(docs)}")

# print(docs[3].page_content)