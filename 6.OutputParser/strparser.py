from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",  # Chat-compatible
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)


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

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser 

# Run the chain

result = chain.invoke({'topic':'Jainism'})

print("Detailed Summary:\n", result)