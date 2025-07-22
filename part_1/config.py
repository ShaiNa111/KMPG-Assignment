import os

from dotenv import load_dotenv
from pathlib import Path


# Load .env variables into environment
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

AZURE_FORM_KEY = os.getenv("AZURE_FORM_KEY")
AZURE_FORM_ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
