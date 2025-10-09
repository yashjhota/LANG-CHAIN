import os
import wikipedia
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()  # loads .env

# Path to store Chroma DB
CHROMA_PATH = Path("chroma_db")
CHROMA_PATH.mkdir(exist_ok=True)

def load_local_docs(folder: str = "data") -> list:
    """Load all .txt and .pdf files from a folder."""
    docs = []
    for file in Path(folder).glob("*"):
        if file.suffix == ".txt":
            loader = TextLoader(str(file))
        elif file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
        else:
            continue
        docs.extend(loader.load())
    return docs

def load_wikipedia_articles(titles: list[str]) -> list:
    """Fetch Wikipedia pages and return as Documents."""
    docs = []
    for title in titles:
        try:
            page = wikipedia.page(title)
            content = page.content
            docs.append(
                Document(page_content=content, metadata={"source": f"Wikipedia:{title}"})
            )
        except Exception as e:
            print(f"Failed to load {title}: {e}")
    return docs

def create_chroma_store(docs: list, persist_path: Path = CHROMA_PATH):
    """Create a Chroma vector store from documents."""
    # Use Groq embeddings (free tier)
    embeddings =HuggingFaceEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2"
    )
    # Persist the store
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(persist_path),
    )
    db.persist()
    print(f"Chroma DB stored at {persist_path}")


if __name__ == "__main__":
    # 1️⃣ Load local docs
    local_docs = load_local_docs()

    # # 2️⃣ Load Wikipedia articles (optional)
    # wiki_titles = ["Python (programming language)", "LangChain", "Groq"]
    # wiki_docs = load_wikipedia_articles(wiki_titles)

    # # 3️⃣ Combine
    # all_ + wiki_docsdocs = local_docs

    # 4️⃣ Create Chroma store
    create_chroma_store(local_docs)