import os
import json
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from typing import Dict, Any

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Research Grapher", page_icon="🚀", layout="wide")

# Best practice: Load API Key from environment variables or Streamlit secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
if not GEMINI_API_KEY:
    st.error("Missing Gemini API Key. Please set the GEMINI_API_KEY environment variable.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = 'gemini-1.5-flash'

# --- CORE LOGIC ---

def parse_pdf(file) -> str:
    """Extracts text from an uploaded PDF file."""
    try:
        reader = PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages])
        return text
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

@st.cache_data(show_spinner=False)
def extract_paper_metadata(text: str) -> Dict[str, Any]:
    """
    Leverages Gemini LLM to extract structured metadata from research text.
    Uses caching to prevent redundant API calls.
    """
    model = genai.GenerativeModel(MODEL_NAME)
    
    # Prompt engineering with clear constraints
    prompt = f"""
    Analyze the provided research paper text. 
    1. Identify the paper's Title and list of Authors.
    2. Extract the first 10 references/citations mentioned.
    
    Return the data strictly in valid JSON format:
    {{
        "title": "string",
        "authors": ["string"],
        "references": [{"title": "string", "year": "integer"}]
    }}

    Text Content (Truncated):
    {text[:16000]} 
    """

    try:
        response = model.generate_content(
            prompt, 
            generation_config={"response_mime_type": "application/json"}
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": f"AI Processing failed: {str(e)}"}

# --- USER INTERFACE ---

def main():
    st.title("Research Grapher 🚀")
    st.markdown("Structure your research citations instantly using Gemini 1.5 Flash.")
    
    with st.sidebar:
        st.header("Upload Center")
        uploaded_file = st.file_uploader("Choose a Research PDF", type="pdf")
        st.info("Note: Only the first 16,000 characters are sent for analysis to optimize speed.")

    if uploaded_file:
        # Displaying the results in organized columns
        col1, col2 = st.columns([1, 1])

        with st.spinner("Processing PDF and querying Gemini..."):
            raw_text = parse_pdf(uploaded_file)
            
            if raw_text:
                data = extract_paper_metadata(raw_text)

                if "error" in data:
                    st.error(data["error"])
                else:
                    with col1:
                        st.subheader("Paper Metadata")
                        st.success(f"**Title:** {data.get('title', 'Unknown')}")
                        st.write(f"**Authors:** {', '.join(data.get('authors', []))}")

                    with col2:
                        st.subheader("Top Citations")
                        if data.get('references'):
                            st.table(data['references'])
                        else:
                            st.warning("No references were found by the AI.")

if __name__ == "__main__":
    main()
