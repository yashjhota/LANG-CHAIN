from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# Using  a FREE HuggingFace model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-1.1-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Define sentiment model output
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

parser2 = PydanticOutputParser(pydantic_object=Feedback)

# Prompt to classify feedback


prompt1 = PromptTemplate(
    template=(
        "Classify the sentiment of this feedback as either 'positive' or 'negative'.\n"
        "Respond with a JSON like this: {{ \"sentiment\": \"positive\" }}\n\n"
        "Feedback: {feedback}"
    ),
    input_variables=["feedback"]
)


# Sentiment classifier chain
classifier_chain = prompt1 | model | parser2

# Positive feedback response
prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive feedback:\n{feedback}',
    input_variables=['feedback']
)

# Negative feedback response
prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative feedback:\n{feedback}',
    input_variables=['feedback']
)

# Conditional branch based on sentiment
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', prompt2 | model | parser),
    (lambda x: x.sentiment == 'negative', prompt3 | model | parser),
    RunnableLambda(lambda x: "❌ Could not determine sentiment.")
)

# Final flow
chain = classifier_chain | branch_chain


print("\n📝 Final Response:\n")

print(chain.invoke({'feedback': 'This is a beautiful phone'}))