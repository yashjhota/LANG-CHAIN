from langchain.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceHubEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()
# Make sure your .env file contains: HUGGINGFACEHUB_API_TOKEN=your_token_here

embedding_model = HuggingFaceHubEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=3
)

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

docs = text_splitter.create_documents([sample])
print(len(docs))
print(docs)
