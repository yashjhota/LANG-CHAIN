from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize the Hugging Face endpoint
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Generate 10 facts about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

# ✅ chain flow: Prompt → Model → Parser
chain = prompt | model | parser

# Run the chain
response = chain.invoke({'topic': "Mahaveer Swami"})
print(response)
# chain.get_graph().print_ascii()

