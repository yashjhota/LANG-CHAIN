# you have to pay for using openai models

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


emd = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    dimensions=32)

result = emd.embed_query("Hello world")

print(str(result))