from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


import streamlit as st
from dotenv import load_dotenv
load_dotenv()


### Prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps people find information."),
        ("user","Question:{question}")
    ]
)

### Streamlit App

st.title("Chat with Groq LLM")
st.write("Ask a question and get an answer from the Groq LLM.")
question = st.text_input("Enter your question:")


### Model Call

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
parser = StrOutputParser()

### Chain

chain = prompt | llm | parser

if question:
    response = chain.invoke({"question": question})
    st.write("Answer:", response)