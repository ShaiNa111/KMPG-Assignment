import json

from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY, AZURE_API_VERSION, OPENAI_ENGINE

llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_API_VERSION,
    model=OPENAI_ENGINE,
    temperature=0,
)


def extract_fields_with_gpt(ocr_text: str) -> dict:
    """
    This function is used to extract the relevant fields with GPT.
    Args:
        ocr_text: Extracted data from the OCR process.

    Returns: JSON of the relevant data

    """
    # prompt_template = """
    # You are an assistant that extracts structured data from National Insurance forms (ביטוח לאומי).
    # Based on the following OCR text, return the result as a valid JSON **without any explanation or formatting**.
    #
    # Return only this structure:
    # {
    #   "lastName": "",
    #   "firstName": "",
    #   "idNumber": "",
    #   "gender": "",
    #   "dateOfBirth": {
    #     "day": "",
    #     "month": "",
    #     "year": ""
    #   },
    #   "address": {
    #     "street": "",
    #     "houseNumber": "",
    #     "entrance": "",
    #     "apartment": "",
    #     "city": "",
    #     "postalCode": "",
    #     "poBox": ""
    #   },
    #   "landlinePhone": "",
    #   "mobilePhone": "",
    #   "jobType": "",
    #   "dateOfInjury": {
    #     "day": "",
    #     "month": "",
    #     "year": ""
    #   },
    #   "timeOfInjury": "",
    #   "accidentLocation": "",
    #   "accidentAddress": "",
    #   "accidentDescription": "",
    #   "injuredBodyPart": "",
    #   "signature": "",
    #   "formFillingDate": {
    #     "day": "",
    #     "month": "",
    #     "year": ""
    #   },
    #   "formReceiptDateAtClinic": {
    #     "day": "",
    #     "month": "",
    #     "year": ""
    #   },
    #   "medicalInstitutionFields": {
    #     "healthFundMember": "",
    #     "natureOfAccident": "",
    #     "medicalDiagnoses": ""
    #   }
    # }
    #
    # If a field is not available, return it as an empty string.
    #
    # OCR TEXT:
    # {ocr_text}
    #
    # """

    prompt_template = PromptTemplate(
        input_variables=['ocr_text'],
        template="""
    You are an assistant that extracts structured data from National Insurance forms (ביטוח לאומי).
    Based on the following OCR text, return the result as a valid JSON **without any explanation or formatting**.

    Return only this structure:
    {{
      "lastName": "",
      "firstName": "",
      "idNumber": "",
      "gender": "",
      "dateOfBirth": {{
        "day": "",
        "month": "",
        "year": ""
      }},
      "address": {{
        "street": "",
        "houseNumber": "",
        "entrance": "",
        "apartment": "",
        "city": "",
        "postalCode": "",
        "poBox": ""
      }},
      "landlinePhone": "",
      "mobilePhone": "",
      "jobType": "",
      "dateOfInjury": {{
        "day": "",
        "month": "",
        "year": ""
      }},
      "timeOfInjury": "",
      "accidentLocation": "",
      "accidentAddress": "",
      "accidentDescription": "",
      "injuredBodyPart": "",
      "signature": "",
      "formFillingDate": {{
        "day": "",
        "month": "",
        "year": ""
      }},
      "formReceiptDateAtClinic": {{
        "day": "",
        "month": "",
        "year": ""
      }},
      "medicalInstitutionFields": {{
        "healthFundMember": "",
        "natureOfAccident": "",
        "medicalDiagnoses": ""
      }}
    }}

    If a field is not available, return it as an empty string.
    
    OCR TEXT:
    {ocr_text}

    """
    )

    response = llm.invoke([{"role": "user", "content": prompt_template.format(ocr_text=ocr_text)}])

    json_output = json.loads(response.content)
    return json_output
