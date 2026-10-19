import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

DB_FOLDER = "faiss_db"

# the "meaning to numbers" model (free with your Google key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# the "cutter"
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def add_pdf(pdf_path: str):
    """Steps 1-4: load, split, embed, store. Returns (pages, chunks)."""
    pages = PyPDFLoader(pdf_path).load()              # 1. LOAD
    chunks = splitter.split_documents(pages)          # 2. SPLIT

    if os.path.exists(DB_FOLDER):                     # 3+4. EMBED + STORE
        store = FAISS.load_local(DB_FOLDER, embeddings, allow_dangerous_deserialization=True)
        store.add_documents(chunks)                   # add to the existing database
    else:
        store = FAISS.from_documents(chunks, embeddings)

    store.save_local(DB_FOLDER)
    return len(pages), len(chunks)


def search_pdf(question: str, k: int = 4):
    """Step 5: find the k chunks closest in meaning to the question."""
    if not os.path.exists(DB_FOLDER):
        return None                                   # nothing uploaded yet
    store = FAISS.load_local(DB_FOLDER, embeddings, allow_dangerous_deserialization=True)
    return store.similarity_search(question, k=k)