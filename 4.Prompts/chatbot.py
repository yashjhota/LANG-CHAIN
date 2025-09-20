from  langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
# Load the Hugging Face model and endpoint
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input.lower() == "exit":
        break
    response = model.invoke(chat_history)
    chat_history.append(response.content)
    print("AI: ", response.content)

    # Optionally, you can print the entire chat history

print("\nChat History:")