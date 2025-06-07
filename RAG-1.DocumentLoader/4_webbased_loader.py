from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


loader = WebBaseLoader("https://type.link/jhotayash")

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro")

prompt = PromptTemplate(
    input_variables=["question","text"],
    template="Answer the following {question} from the  - \n {text}",)



parser = StrOutputParser()

chain = prompt | model | parser

docs = loader.load()

result = chain.invoke({
    'question':"What are the skills?",
    'text':docs[0].page_content
})

print(result)
print("---------------------------------------------------------")
