import os
import json
import streamlit as st
import google.generativeai as genai
import networkx as nx
import streamlit.components.v1 as components
from PyPDF2 import PdfReader
from pyvis.network import Network
from typing import Dict, Any, List

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Research Grapher", page_icon="🚀", layout="wide")

# Best practice: Load API Key from environment variables
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
    """Leverages Gemini LLM to extract structured metadata from research text."""
    model = genai.GenerativeModel(MODEL_NAME)
    
    # Prompt engineering to ensure JSON output
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

def generate_network_graph(main_title: str, references: List[Dict[str, Any]]):
    """Generates an interactive Pyvis network graph."""
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    
    # Add central node (the paper)
    net.add_node(main_title, label=main_title, color="#97c2fc", size=30)
    
    # Add reference nodes and edges
    for ref in references:
        ref_title = ref.get('title', 'Unknown Ref')
        net.add_node(ref_title, label=ref_title, color="#fb7e81", size=15)
        net.add_edge(main_title, ref_title)
        
    # Generate HTML string
    return net.generate_html()

# --- USER INTERFACE ---

def main():
    st.title("Research Grapher 🚀")
    st.markdown("Structure your research citations instantly using Gemini 1.5 Flash.")
    
    with st.sidebar:
        st.header("Upload Center")
        uploaded_file = st.file_uploader("Choose a Research PDF", type="pdf")
        st.info("Analysis covers the first 16,000 characters.")

    if uploaded_file:
        with st.spinner("Processing PDF and querying Gemini..."):
            raw_text = parse_pdf(uploaded_file)
            
            if raw_text:
                data = extract_paper_metadata(raw_text)

                if "error" in data:
                    st.error(data["error"])
                else:
                    # Layout Results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader("Paper Metadata")
                        st.success(f"**Title:** {data.get('title', 'Unknown')}")
                        st.write(f"**Authors:** {', '.join(data.get('authors', []))}")
                        
                        st.subheader("Reference List")
                        st.table(data['references'])

                    with col2:
                        st.subheader("Network Visualization")
                        # Generate graph
                        html_graph = generate_network_graph(data.get('title', 'Paper'), data.get('references', []))
                        
                        # Render Pyvis HTML in Streamlit
                        components.html(html_graph, height=600)

if __name__ == "__main__":
    main()
