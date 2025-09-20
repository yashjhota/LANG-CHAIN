from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain about the joke on {txt}",
    input_variables=["txt"]

)

output_parser = StrOutputParser()

chain = RunnableSequence(prompt1,model,output_parser,prompt2,model,output_parser)

result = chain.invoke({"topic": "cats"})
print(result) 


chain.get_graph().print_ascii()