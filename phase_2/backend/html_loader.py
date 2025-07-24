from pathlib import Path
from typing import List

from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_knowledgebase(folder_path="../data") -> List[Document]:
    """
    This function is used to load the html files into list of Documents to save them using Vector DB called FAISS to find semantic meaning later.
    Args:
        folder_path:

    Returns:

    """

    knowledge_base_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for html_file in Path(folder_path).glob("*.html"):
        loader = UnstructuredHTMLLoader(html_file)
        doc = loader.load()
        split_docs = text_splitter.split_documents(doc)
        knowledge_base_docs.extend(split_docs)

    return knowledge_base_docs
