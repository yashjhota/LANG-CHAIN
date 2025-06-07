from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta", 
    task="text-generation",
    
)

# Step 2: Wrap the LLM into ChatHuggingFace
chat_model = ChatHuggingFace(llm=llm)

# Step 3: Use the chat model
response = chat_model.invoke("What is the capital of India?")
print(response.content)
