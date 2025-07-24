import streamlit as st
import sys
import os
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BACKEND_URL


st.set_page_config(page_title="HMO Chatbot", layout="centered")
st.title("Medical HMO Chatbot")

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "phase" not in st.session_state:
    st.session_state.phase = "collect_data"

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    if st.session_state.phase == "collect_data":
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            response = requests.post(f"{BACKEND_URL}/chat", json={
                "messages": st.session_state.messages,
                "user_prompt": user_input
            }).json()

            res_content = response.get('content')
            st.session_state.messages.append({"role": "assistant", "content": res_content})

        with st.chat_message("assistant"):
            st.write(res_content)

        user_info = response.get('user_info', {})
        missing_fields = response.get('missing_fields', [])
        if user_info.get('is_confirmed', False) and not missing_fields:
            st.session_state.phase = 'qa'
            st.session_state.user_info = user_info
    else:
        with st.chat_message("user"):
            st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Thinking..."):
            response = requests.post(f"{BACKEND_URL}/qa", json={
                "user_prompt": user_input,
                "user_info": st.session_state.user_info,
            }).json()

            response_content = response.get('content')
        with st.chat_message("assistant"):
            st.write(response_content)
