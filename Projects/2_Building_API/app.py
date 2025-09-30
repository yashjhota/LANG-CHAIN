from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import uvicorn
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Building API with Groq and LangChain",
    description="An API built using FastAPI, Groq, and LangChain.",
    version="0.1.0",
)

model = ChatGroq(model="openai/gpt-oss-20b")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("user", "{text}"),
    ]
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Groq and LangChain API!"}

@app.put("/translate")
def translate(text: str, input_language: str = "English", output_language: str = "French"):
    response = model.predict_messages(
        prompt.format_messages(
            text=text,
            input_language=input_language,
            output_language=output_language
        )
    )
    return {"translated_text": response.content}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
