from  langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

# Create a Streamlit app to interact with the model
st.title("Chat with TinyLlama")
st.header("Research Tool")
input = st.text_input("Enter your question:")

if st.button("Summarize"):
    result = model.invoke(input)
    st.write(result.content)


