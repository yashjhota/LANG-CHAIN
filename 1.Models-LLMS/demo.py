# You dont need to study llms more , focus on chat models which are fine tuned on LLMS.

from langchain.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-4o-mini")

result = llm.invoke("What is the capital of India?")

print(result)