from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
import logging
import os
import json
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables for vector store
vector_store = None
qa_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_knowledge_base()
    yield
    # Shutdown
    logger.info("Shutting down application")


app = FastAPI(
    title="Israeli Health Funds Chatbot API",
    description="Microservice for handling medical services queries",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class UserInfo(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    id_number: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    hmo_name: Optional[str] = None
    hmo_card_number: Optional[str] = None
    insurance_tier: Optional[str] = None
    is_confirmed: bool = False

    @validator('id_number')
    def validate_id_number(cls, v):
        if v and (not v.isdigit() or len(v) != 9):
            raise ValueError('ID number must be exactly 9 digits')
        return v

    @validator('age')
    def validate_age(cls, v):
        if v is not None and (v < 0 or v > 120):
            raise ValueError('Age must be between 0 and 120')
        return v

    @validator('hmo_name')
    def validate_hmo_name(cls, v):
        if v and v not in ['מכבי', 'מאוחדת', 'כללית', 'Maccabi', 'Meuhedet', 'Clalit']:
            raise ValueError('Invalid HMO name')
        return v

    @validator('insurance_tier')
    def validate_insurance_tier(cls, v):
        if v and v not in ['זהב', 'כסף', 'ארד', 'Gold', 'Silver', 'Bronze']:
            raise ValueError('Invalid insurance tier')
        return v


class ConversationMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = datetime.now()


class ChatRequest(BaseModel):
    message: str
    user_info: UserInfo
    conversation_history: List[ConversationMessage] = []
    language: str = "he"  # Hebrew by default


class ChatResponse(BaseModel):
    response: str
    user_info: UserInfo
    conversation_history: List[ConversationMessage]
    phase: str  # 'collection' or 'qa'
    status: str = "success"


# Azure OpenAI configuration
async def get_azure_openai_client():
    try:
        client = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview",
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4"),
            temperature=0.7
        )
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Azure OpenAI client: {e}")
        raise HTTPException(status_code=500, detail="AI service initialization failed")


# Knowledge base initialization
async def initialize_knowledge_base():
    global vector_store, qa_chain
    try:
        logger.info("Initializing knowledge base...")

        # Load HTML files
        documents = []
        html_files = [
            'phase2_data/alternative_services.html',
            'phase2_data/communication_clinic_services.html',
            'phase2_data/dentel_services.html',
            'phase2_data/optometry_services.html',
            'phase2_data/pragrency_services.html',
            'phase2_data/workshops_services.html'
        ]

        for file_path in html_files:
            if os.path.exists(file_path):
                loader = UnstructuredHTMLLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {file_path}")

        if not documents:
            logger.warning("No documents loaded from HTML files")
            return

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-15-preview"
        )

        vector_store = FAISS.from_documents(splits, embeddings)

        # Create QA chain
        llm = await get_azure_openai_client()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        logger.info("Knowledge base initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize knowledge base: {e}")
        raise


# Prompt templates
COLLECTION_PROMPT_HE = """
אתה עוזר AI שמיועד לאסוף מידע מהמשתמש עבור קופת חולים בישראל.
המטרה שלך היא לאסוף את המידע הבא באופן טבעי ונעים:
- שם פרטי ושם משפחה
- מספר תעודת זהות (9 ספרות)
- מין
- גיל (בין 0 ל-120)
- שם קופת החולים (מכבי, מאוחדת, כללית)
- מספר כרטיס קופת חולים (9 ספרות)
- דרגת ביטוח (זהב, כסף, ארד)

המידע הנוכחי של המשתמש:
שם פרטי: {first_name}
שם משפחה: {last_name}
תעודת זהות: {id_number}
מין: {gender}
גיל: {age}
קופת חולים: {hmo_name}
מספר כרטיס: {hmo_card_number}
דרגת ביטוח: {insurance_tier}

הנחיות:
1. בקש מידע חסר באופן טבעי
2. אמת שהמידע תקין
3. לאחר השלמת כל המידע, הצג סיכום ובקש אישור
4. היה נעים וידידותי
5. דבר בעברית

היסטוריית השיחה:
{conversation_history}

הודעת המשתמש: {user_message}
"""

COLLECTION_PROMPT_EN = """
You are an AI assistant designed to collect user information for Israeli health funds.
Your goal is to collect the following information naturally and pleasantly:
- First and last name
- ID number (9 digits)
- Gender
- Age (between 0 and 120)
- HMO name (Maccabi, Meuhedet, Clalit)
- HMO card number (9 digits)
- Insurance tier (Gold, Silver, Bronze)

Current user information:
First name: {first_name}
Last name: {last_name}
ID number: {id_number}
Gender: {gender}
Age: {age}
HMO name: {hmo_name}
Card number: {hmo_card_number}
Insurance tier: {insurance_tier}

Guidelines:
1. Request missing information naturally
2. Validate that information is correct
3. After completing all information, show summary and ask for confirmation
4. Be pleasant and friendly
5. Communicate in English

Conversation history:
{conversation_history}

User message: {user_message}
"""

QA_PROMPT_HE = """
אתה עוזר AI שמתמחה במידע על שירותי קופות החולים בישראל.
אתה עונה על שאלות בהתבסס על המידע הרלוונטי מבסיס הנתונים.

מידע המשתמש:
שם: {first_name} {last_name}
קופת חולים: {hmo_name}
דרגת ביטוח: {insurance_tier}
גיל: {age}

הנחיות:
1. ענה על שאלות בהתבסס על המידע הרלוונטי לקופת החולים ודרגת הביטוח של המשתמש
2. אם המידע לא קיים, הודע על כך בבירור
3. היה מדויק ומועיל
4. דבר בעברית

מידע רלוונטי: {context}

שאלת המשתמש: {question}
"""

QA_PROMPT_EN = """
You are an AI assistant specializing in information about Israeli health fund services.
You answer questions based on relevant information from the database.

User information:
Name: {first_name} {last_name}
HMO: {hmo_name}
Insurance tier: {insurance_tier}
Age: {age}

Guidelines:
1. Answer questions based on information relevant to the user's HMO and insurance tier
2. If information doesn't exist, clearly state this
3. Be accurate and helpful
4. Communicate in English

Relevant information: {context}

User question: {question}
"""


async def process_collection_phase(request: ChatRequest) -> ChatResponse:
    """Process user information collection phase"""
    try:
        client = await get_azure_openai_client()

        # Select prompt based on language
        prompt_template = COLLECTION_PROMPT_HE if request.language == "he" else COLLECTION_PROMPT_EN

        # Format conversation history
        history_text = "\n".join([
            f"{msg.role}: {msg.content}" for msg in request.conversation_history[-5:]  # Last 5 messages
        ])

        # Format prompt
        prompt = prompt_template.format(
            first_name=request.user_info.first_name or "לא מצוין" if request.language == "he" else "Not specified",
            last_name=request.user_info.last_name or "לא מצוין" if request.language == "he" else "Not specified",
            id_number=request.user_info.id_number or "לא מצוין" if request.language == "he" else "Not specified",
            gender=request.user_info.gender or "לא מצוין" if request.language == "he" else "Not specified",
            age=request.user_info.age or "לא מצוין" if request.language == "he" else "Not specified",
            hmo_name=request.user_info.hmo_name or "לא מצוין" if request.language == "he" else "Not specified",
            hmo_card_number=request.user_info.hmo_card_number or "לא מצוין" if request.language == "he" else "Not specified",
            insurance_tier=request.user_info.insurance_tier or "לא מצוין" if request.language == "he" else "Not specified",
            conversation_history=history_text,
            user_message=request.message
        )

        # Get AI response
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=request.message)
        ]

        response = await client.agenerate([messages])
        ai_response = response.generations[0][0].text

        # Update conversation history
        updated_history = request.conversation_history + [
            ConversationMessage(role="user", content=request.message),
            ConversationMessage(role="assistant", content=ai_response)
        ]

        # Extract user information from the conversation (simplified logic)
        # In a real implementation, you might use NLP to extract this information
        updated_user_info = request.user_info

        # Determine if we should move to QA phase
        phase = "collection"
        if updated_user_info.is_confirmed and all([
            updated_user_info.first_name,
            updated_user_info.last_name,
            updated_user_info.id_number,
            updated_user_info.hmo_name,
            updated_user_info.insurance_tier
        ]):
            phase = "qa"

        return ChatResponse(
            response=ai_response,
            user_info=updated_user_info,
            conversation_history=updated_history,
            phase=phase
        )

    except Exception as e:
        logger.error(f"Error in collection phase: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_qa_phase(request: ChatRequest) -> ChatResponse:
    """Process Q&A phase with knowledge base"""
    try:
        if not qa_chain:
            raise HTTPException(status_code=500, detail="Knowledge base not initialized")

        # Select prompt based on language
        prompt_template = QA_PROMPT_HE if request.language == "he" else QA_PROMPT_EN

        # Query the knowledge base
        result = qa_chain({"query": request.message})

        # Format the response with user context
        context_prompt = prompt_template.format(
            first_name=request.user_info.first_name,
            last_name=request.user_info.last_name,
            hmo_name=request.user_info.hmo_name,
            insurance_tier=request.user_info.insurance_tier,
            age=request.user_info.age,
            context=result["result"],
            question=request.message
        )

        client = await get_azure_openai_client()
        messages = [
            SystemMessage(content=context_prompt),
            HumanMessage(content=request.message)
        ]

        response = await client.agenerate([messages])
        ai_response = response.generations[0][0].text

        # Update conversation history
        updated_history = request.conversation_history + [
            ConversationMessage(role="user", content=request.message),
            ConversationMessage(role="assistant", content=ai_response)
        ]

        return ChatResponse(
            response=ai_response,
            user_info=request.user_info,
            conversation_history=updated_history,
            phase="qa"
        )

    except Exception as e:
        logger.error(f"Error in QA phase: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        logger.info(f"Received chat request: {request.message[:50]}...")

        # Determine current phase
        if not request.user_info.is_confirmed:
            response = await process_collection_phase(request)
        else:
            response = await process_qa_phase(request)

        logger.info(f"Generated response successfully, phase: {response.phase}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/confirm_user_info")
async def confirm_user_info(user_info: UserInfo):
    """Confirm user information"""
    try:
        user_info.is_confirmed = True
        logger.info(f"User info confirmed for: {user_info.first_name} {user_info.last_name}")

        return {
            "status": "success",
            "message": "User information confirmed",
            "user_info": user_info
        }
    except Exception as e:
        logger.error(f"Error confirming user info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Israeli Health Funds Chatbot API",
        "version": "1.0.0",
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
