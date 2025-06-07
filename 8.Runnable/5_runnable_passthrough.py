from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro")

prompt1 = PromptTemplate(
    input_variables=["text"],
    template="write a tweet about {text}"
)

prompt2 = PromptTemplate(
    input_variables=["text"],
    template="write a summary of {text}"
)

parser = StrOutputParser()

joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel(
    {
        "tweet": RunnablePassthrough(),
        "summary": RunnableSequence(prompt2, model, parser)
    }
)

final_chain = RunnableSequence(
    joke_chain,
    parallel_chain
)

result = final_chain.invoke("Python!")

print(result)