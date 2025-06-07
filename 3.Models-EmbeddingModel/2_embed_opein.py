from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


emd = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    dimensions=32)

documents = [
    "Hello world",
    "Goodbye world",
    "Hello again",
    "Goodbye again",
]
result = emd.embed_documents(documents)

print(str(result))