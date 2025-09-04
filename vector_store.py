import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

def load_vectorstore(vector_store_path):
    embeddings = OllamaEmbeddings(model="mistral")
    return FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
