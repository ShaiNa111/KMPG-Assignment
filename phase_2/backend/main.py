import os
import sys
import nltk
import ssl

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from vector_store_loader import load_vector_store_once
from phase_2.llm_client import extract_user_info_with_gpt, get_qa_chain_response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function is used to runs only once, it downloads nltk package and load the knownladge base into FAISS database.
    Args:
        app:

    Returns:

    """
    # Disable SSL certificate verification (only if certifi fails)
    ssl._create_default_https_context = ssl._create_unverified_context
    nltk.download('punkt')
    load_vector_store_once()
    yield

app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def collect_data(request: Request):
    data = await request.json()
    user_prompt = data.get("user_prompt", {})
    messages = data.get("messages", [])
    res = extract_user_info_with_gpt(user_prompt, messages)
    return res


@app.post("/qa")
async def qa_phase(request: Request):
    data = await request.json()
    user_info = data.get("user_info", {})
    user_prompt = data.get("user_prompt", {})
    content = get_qa_chain_response(user_prompt, user_info)
    return {"content": content}
