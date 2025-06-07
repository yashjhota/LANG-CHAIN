from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# just to save the model on different folder

# os.environ['HF_HOME'] ='D:/HuggingFace'

llm= HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    model_kwargs={"temperature": 0.7, "max_length": 512}
)
model = ChatHuggingFace(
    llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)