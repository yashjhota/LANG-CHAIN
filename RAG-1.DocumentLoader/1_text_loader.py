from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

loader = TextLoader("poem.txt")

docs = loader.load()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro")

prompt = PromptTemplate(
    input_variables=["text"],
    template="write a summary of the following {text}")



parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({'text':docs[0].page_content}))

# print(type(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)

