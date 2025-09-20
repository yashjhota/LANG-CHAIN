from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

# Use two different HuggingFace endpoints for demo (you can use the same one twice if needed)
llm1 = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)

llm2 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation"
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

# Prompts
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text:\n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document:\nnotes -> {notes}\nquiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

# Parallel chain (note & quiz generation)
parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

# Merge step
merge_chain = prompt3 | model1 | parser

# Final pipeline
chain = parallel_chain | merge_chain

# Sample text
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

Advantages:
- Effective in high dimensional spaces.
- Memory efficient.
- Versatile kernel functions.

Disadvantages:
- Risk of overfitting.
- No direct probability estimates.

SVMs in scikit-learn support both dense and sparse data.
"""

# Invoke the chain
result = chain.invoke({'text': text})
print("\n📄 Final Merged Output:\n")
print(result)
