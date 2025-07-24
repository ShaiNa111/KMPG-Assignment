import json
import logging
import traceback

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain.chains import create_retrieval_chain

from phase_2.backend.models import UserInfo
from phase_2.backend.prompt_templates import PromptTemplates
from phase_2.backend.vector_store_loader import load_vector_store_once
from config import AZURE_API_VERSION, AZURE_OPENAI_KEY, AZURE_OPENAI_ENDPOINT, OPENAI_ENGINE



def extract_user_info_with_gpt(user_prompt, messages: list) -> UserInfo:
    """
    This function is used to extract user info with gpt.

    Returns: Dict["context": {}, "user_info": {}, "missing_fields": []]
    """
    response_format = {
        "content": "",
        "user_info": {},
        "missing_fields": []
    }

    info_prompt_template = PromptTemplate(
        input_variables=['user_prompt'],
        template=PromptTemplates.INFO_COLLECTION_SYSTEM_PROMPT
    )
    messages.append({"role": "user", "content": info_prompt_template.format(user_prompt=user_prompt)})

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_API_VERSION,
        model=OPENAI_ENGINE,
        temperature=0,
    )

    response = llm.invoke(messages)

    # Attempt to parse the response content as JSON
    try:
        response_format = json.loads(response.content)
    except json.JSONDecodeError:
        logging.error("Cannot load the response content", traceback.format_exc())
    return response_format


def get_qa_chain_response(user_prompt, user_info: dict):
    """
    This function is used to get the QA response from GPT,
    Args:
        user_info:

    Returns:

    """
    vector_store = load_vector_store_once()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Retrieve relevant documents based on user question to add context as knowledge base in my prompt
    docs = retriever.get_relevant_documents(user_prompt)

    knowledge_content = "\n\n".join([doc.page_content for doc in docs])
    customize_prompt = PromptTemplates.get_qa_prompt(user_info=UserInfo(**user_info), knowledge_content=knowledge_content, user_prompt=user_prompt)

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_API_VERSION,
        model=OPENAI_ENGINE,
        temperature=0.3,
    )

    response = llm.invoke(customize_prompt)

    return response.content
