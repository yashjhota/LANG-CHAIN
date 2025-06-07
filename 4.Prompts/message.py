from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from  langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
# Load the Hugging Face model and endpoint
llm = HuggingFaceEndpoint(
    repo_id="tiiuae/falcon-7b-instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


messages=[
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)
