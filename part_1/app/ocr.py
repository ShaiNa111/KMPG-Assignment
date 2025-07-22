
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from part_1.config import AZURE_FORM_ENDPOINT, AZURE_FORM_KEY

# Connect to Azure document form recognizer by the specifiy credentials
client = DocumentAnalysisClient(
    endpoint=AZURE_FORM_ENDPOINT,
    credential=AzureKeyCredential(AZURE_FORM_KEY)
)


def extract_text_from_file(file) -> str:
    """
    This function extract the data from a given file.
    Args:
        file: PDF/JPG

    Returns: string that represent the result of the OCR process
    """
    poller = client.begin_analyze_document("prebuilt-layout", document=file)
    result = poller.result()

    return result.content
