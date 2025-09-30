import streamlit as st
import requests

st.title("Client Application")
st.write("This is a simple client application to interact with the Groq and LangChain API.")
text = st.text_area("Enter text to translate", "Hello, how are you?")
input_language = st.text_input("Input Language", "English")
output_language = st.text_input("Output Language", "French")

if st.button("Translate"):
    response = requests.put(
        "http://localhost:8000/translate",
        params={
            "text": text,
            "input_language": input_language,
            "output_language": output_language
        }
    )
    if response.status_code == 200:
        translated_text = response.json().get("translated_text")
        st.success(f"Translated Text: {translated_text}")
    else:
        st.error("Error in translation request.")

# To run the client application, use the command:
# streamlit run client.py