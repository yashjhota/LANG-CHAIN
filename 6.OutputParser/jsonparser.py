from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",  # Chat-compatible
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)


parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name , age , city of an fictional person \n {format_instructions} ",
    input_variables=[""],
    partial_variables={"format_instructions": parser.get_format_instructions()}
    # partial variables are filled before runtime not at runtime
)

prompt = template.format()

print(prompt)

result = model.invoke(prompt)

final = parser.parse(result.content)

print(result)

print(final)


"""
    Biggest problem is that you can enforce any schema like you cant decide the structuring of output
"""