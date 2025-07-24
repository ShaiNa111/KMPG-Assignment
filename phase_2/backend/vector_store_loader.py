import logging

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from html_loader import load_knowledgebase
from config import AZURE_API_VERSION, AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT

# Global object
VECTOR_STORE = None


def load_vector_store_once():
    global VECTOR_STORE
    if VECTOR_STORE is None:
        logging.info("Loading HTML documents and creating vector store...")
        docs = load_knowledgebase()
        embedding_model = AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_API_VERSION)
        VECTOR_STORE = FAISS.from_documents(docs, embedding=embedding_model)
        logging.info("Vector store loaded.")
    return VECTOR_STORE
