from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize the Hugging Face endpoint
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Generate 5 facts about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

# ✅ Correct chain flow: Prompt → Model → Parser
chain = prompt | model | parser

# Run the chain
response = chain.invoke({'topic': "Python programming"})
print(response)
chain.get_graph().print_ascii()

