from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq


load_dotenv()

class Review(BaseModel):
    summary: str = Field(..., description="summary of the review")
    sentiment: str = Field(..., description="sentiment of the review positive/negative")
   


model = ChatGroq(model="llama-3.1-8b-instant")

structured_model = model.with_structured_output(Review)

response = structured_model.invoke("""Tried the new AI tool – super smooth and efficient!
It generates quality content in seconds and saves tons of time.
UI is clean, responses are accurate.
Perfect for students, creators, and busy pros! ⭐⭐⭐⭐⭐""")

print(response)

