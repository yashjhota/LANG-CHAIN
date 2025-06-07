from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate # prompt dynamic  

load_dotenv()

model = ChatGroq(model="llama-3.1-8b-instant")

# Define the prompt 1 > detailed summary

template1=PromptTemplate(
    template="Please summarize the following text in detail:\n\n{topic}\n\nSummary:",
    input_variables=["topic"]
)

# Define the prompt 2 > short summary

template2=PromptTemplate(
    template="write the five line summary of the following text:\n\n{text}\n\nSummary:",
    input_variables=["text"]
)

prompt1= template1.invoke({'topic':'Jainism'})

result = model.invoke(prompt1)

prompt2= template2.invoke({'text':result.content})

result2 = model.invoke(prompt2)

print("Detailed Summary:\n", result.content)