from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    input_variables=["text"],
    template="write a tweet about {text}"
)

prompt2 = PromptTemplate(
    input_variables=["text"],
    template="write a summary of {text}"
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet':RunnableSequence(prompt1,model,parser),
    'summary':RunnableSequence(prompt2,model,parser)
    })

result = parallel_chain.invoke("AI")

# print(result['tweet'])
# print(result['summary'])

parallel_chain.get_graph().print_ascii()