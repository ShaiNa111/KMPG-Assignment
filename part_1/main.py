import streamlit as st

from app.extractor import extract_fields_with_gpt
from app.ocr import extract_text_from_file


st.title("National Insurance Institute Form Extractor (ביטוח לאומי)")
st.write("Upload a form (PDF or JPG) and extract structured data.")

uploaded_file = st.file_uploader("Choose a file", type=["pdf", "jpg", "jpeg"])

if uploaded_file:
    with st.spinner("Processing OCR..."):
        ocr_text = extract_text_from_file(uploaded_file)

    st.subheader("OCR Text Preview")
    st.text_area("OCR Output", ocr_text, height=200)

    with st.spinner("Extracting fields using GPT..."):
        extracted_json = extract_fields_with_gpt(ocr_text)

    st.subheader("Extracted JSON")
    st.json(extracted_json)
