import os
from dotenv import load_dotenv

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential


# Load .env variables into environment
load_dotenv()

AZURE_FORM_KEY = os.getenv("AZURE_FORM_KEY")
AZURE_FORM_ENDPOINT = os.getenv("AZURE_FORM_ENDPOINT")

# Connect to Azure document form recognizer by the specifiy credentials
client = DocumentAnalysisClient(
    endpoint="https://eastus.api.cognitive.microsoft.com/",
    credential=AzureKeyCredential("efbe39573b3b4e68832a5d4f6d6a391a")
)


def get_pdf_data(pdf_file):
    with open(pdf_file, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-layout", document=f)
        result = poller.result()
    return result
