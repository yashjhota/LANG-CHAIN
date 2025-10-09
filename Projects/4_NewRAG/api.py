from fastapi import FastAPI
from pydantic import BaseModel
from rag_cahin import ask

app = FastAPI(title="RAG API", description="API for Retrieval-Augmented Generation using Groq LLMs and Chroma vector store.", version="1.0.0")

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask", response_model=dict)
def get_answer(request: QuestionRequest):
    """Endpoint to get answer and sources for a given question."""
    result = ask(request.question)
    return result