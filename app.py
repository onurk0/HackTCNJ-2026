import os
import json
import certifi
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pyvis.network import Network
from typing import Dict, Any, List, Optional
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# --- INITIALIZATION ---
load_dotenv()
st.set_page_config(page_title="Research Grapher", page_icon="🚀", layout="wide")

# --- CONSTANTS ---
MODEL_NAME = 'gemini-1.5-flash'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# --- RESOURCE MANAGEMENT ---

@st.cache_resource
def get_ai_client() -> genai.Client:
    """Initializes the Google GenAI Client as a cached resource."""
    if not GEMINI_API_KEY:
        st.error("Missing GEMINI_API_KEY in environment.")
        st.stop()
    return genai.Client(api_key=GEMINI_API_KEY)

@st.cache_resource
def get_mongodb_collection():
    """Establishes and caches the MongoDB connection."""
    if not MONGO_URI:
        st.error("Missing MONGO_URI in environment.")
        st.stop()
    try:
        client = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        # Verify connection
        client.admin.command('ping')
        return client["hackathon_db"]["papers"]
    except PyMongoError as e:
        st.error(f"Database connection failed: {e}")
        st.stop()

# --- CORE LOGIC ---

def parse_pdf(file) -> str:
    """Extracts text from an uploaded PDF file safely."""
    try:
        reader = PdfReader(file)
        # Extract and join text from all pages
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text.strip()
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

@st.cache_data(show_spinner=False)
def extract_paper_metadata(text: str) -> Dict[str, Any]:
    """Uses AI to extract structured metadata from raw text."""
    client = get_ai_client()
    
    # Prompt engineering for strict JSON output
    prompt = (
        "Analyze the following research paper text. "
        "Return a JSON object with keys: 'title' (string), 'authors' (list of strings), "
        "and 'references' (list of objects with 'title' and 'year').\n\n"
        f"Text: {text[:16000]}"
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": f"AI Processing failed: {str(e)}"}

def save_to_db(data: Dict[str, Any]):
    """Persists the extracted metadata to MongoDB."""
    papers_col = get_mongodb_collection()
    try:
        # Avoid saving if it's an error dict
        if "error" not in data:
            papers_col.update_one(
                {"title": data.get("title")}, 
                {"$set": data}, 
                upsert=True
            )
    except PyMongoError as e:
        st.warning(f"Metadata extracted but failed to save to DB: {e}")

def generate_network_graph(main_title: str, references: List[Dict[str, Any]]) -> str:
    """Generates an interactive Pyvis network graph HTML string."""
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add central node
    net.add_node(main_title, label=main_title, color="#97c2fc", size=30, title="Main Paper")
    
    # Add and link reference nodes
    for ref in references:
        ref_title = ref.get('title', 'Unknown Reference')
        net.add_node(ref_title, label=ref_title[:30] + "...", color="#fb7e81", size=15)
        net.add_edge(main_title, ref_title)
        
    return net.generate_html()

# --- USER INTERFACE ---

def render_ui():
    """Main UI rendering logic."""
    st.title("Research Grapher 🚀")
    st.markdown(f"Structure your research citations instantly using **{MODEL_NAME}**.")
    
    with st.sidebar:
        st.header("Upload Center")
        uploaded_file = st.file_uploader("Choose a Research PDF", type="pdf")
        st.divider()
        st.info("💡 Analysis covers the first 16,000 characters of the document.")

    if uploaded_file:
        with st.spinner("Analyzing document structure..."):
            raw_text = parse_pdf(uploaded_file)
            
            if not raw_text:
                st.warning("The PDF appears to be empty or unreadable.")
                return

            data = extract_paper_metadata(raw_text)

            if "error" in data:
                st.error(data["error"])
            else:
                # Save results to MongoDB
                save_to_db(data)
                
                # Layout Results
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("📄 Paper Metadata")
                    st.success(f"**Title:** {data.get('title', 'Unknown')}")
                    st.write(f"**Authors:** {', '.join(data.get('authors', ['Not identified']))}")
                    
                    st.subheader("📚 Key References")
                    st.dataframe(data.get('references', []), use_container_width=True)

                with col2:
                    st.subheader("🕸️ Citation Network")
                    html_graph = generate_network_graph(
                        data.get('title', 'Target Paper'), 
                        data.get('references', [])
                    )
                    components.html(html_graph, height=600)

if __name__ == "__main__":
    render_ui()
