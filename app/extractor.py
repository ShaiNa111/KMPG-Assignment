import os
import openai
import json

from app.ocr import get_pdf_data


AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Set API Key & Endpoint
openai.api_type = "azure"
openai.api_version = "2024-03-01-preview"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_key = AZURE_OPENAI_KEY


def get_pdf_content_as_json(pdf_file: str) -> dict:
    ocr_text = get_pdf_data(pdf_file)
    prompt = f"""
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

    response = openai.ChatCompletion.create(
        engine="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    extracted_data = response["choices"][0]["message"]["content"]
    json_output = json.loads(extracted_data)
    return json_output


get_pdf_content_as_json("../data/phase1_data/283_ex1.pdf")
