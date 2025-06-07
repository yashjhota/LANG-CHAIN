from  langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
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


