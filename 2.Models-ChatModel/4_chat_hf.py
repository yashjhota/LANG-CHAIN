from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.messages import HumanMessage,SystemMessage


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",  # let Hugging Face choose the best provider for you
)

chat_model = ChatHuggingFace(llm=llm)



# messages = [
#     SystemMessage(content="You're a helpful assistant"),
#     HumanMessage(
#         content="What happens when an unstoppable force meets an immovable object?"
#     ),
# ]

messages ="Write a poem about a lonely computer"

ai_msg = chat_model.invoke(messages)
print(ai_msg.content)