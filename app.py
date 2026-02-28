import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import json
import os

# 1. Setup Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel('gemini-1.5-flash')

def extract_references(text):
    prompt = f"""
    Analyze the following research paper text. 
    1. Identify the paper's Title and Authors.
    2. Extract a list of the first 10 citations/references mentioned.
    Return the data strictly in this JSON format:
    {{
        "title": "string",
        "authors": ["string"],
        "references": [{"title": "string", "year": "int"}]
    }}
    Text: {text[:15000]} # Limiting text to stay within fast processing limits
    """
    response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
    return json.loads(response.text)

# 2. Streamlit UI
st.title("Research Grapher 🚀")
uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Analyzing paper with Gemini..."):
        # Read PDF
        reader = PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()
        
        # Parse with AI
        data = extract_references(full_text)
        st.write(f"### Found: {data['title']}")
        
        # Display as a basic list (Graph logic comes next)
        st.table(data['references'])
