"""
    This script demonstrates how to use the HuggingFaceEmbeddings class from the
      langchain_huggingface module to compute embeddings for a set of documents and a query. 
      It then calculates the cosine similarity between the query embedding and the document embeddings 
      to find the most similar document.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about Ms Dhoni'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

""" Calculate cosine similarity between the query and document embeddings """
scores = cosine_similarity([query_embedding], doc_embeddings)[0]   # You have to pass always a 2D array to the cosine_similarity function

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)



