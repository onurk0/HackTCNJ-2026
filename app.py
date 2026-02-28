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

# --- CONFIGURATION ---
load_dotenv()
st.set_page_config(page_title="Research Grapher", page_icon="🚀", layout="wide")

class ResearchManager:
    """Handles Data Extraction, AI Processing, and Database Persistence."""
    
    def __init__(self):
        self.model_name = 'gemini-1.5-flash'
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.mongo_uri = os.getenv("MONGO_URI")

    @st.cache_resource(_self)
    def get_ai_client(_self) -> genai.Client:
        if not _self.api_key:
            st.error("Missing GEMINI_API_KEY in environment.")
            st.stop()
        return genai.Client(api_key=_self.api_key)

    @st.cache_resource(_self)
    def get_db_collection(_self):
        if not _self.mongo_uri:
            st.error("Missing MONGO_URI in environment.")
            st.stop()
        try:
            client = MongoClient(_self.mongo_uri, tlsCAFile=certifi.where())
            client.admin.command('ping')
            return client["hackathon_db"]["papers"]
        except PyMongoError as e:
            st.error(f"Database connection failed: {e}")
            st.stop()

    def parse_pdf(self, file) -> str:
        try:
            reader = PdfReader(file)
            text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
            return text.strip()
        except Exception as e:
            st.error(f"Failed to read PDF: {e}")
            return ""

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        client = self.get_ai_client()
        prompt = (
            "Analyze the following research paper text. "
            "Return a JSON object with keys: 'title' (string), 'authors' (list of strings), "
            "and 'references' (list of objects with 'title' and 'year').\n\n"
            f"Text: {text[:16000]}"
        )
        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            return json.loads(response.text)
        except Exception as e:
            return {"error": f"AI Processing failed: {str(e)}"}

    def save_paper(self, data: Dict[str, Any]):
        if "error" in data or not data.get("title"):
            return
        
        col = self.get_db_collection()
        try:
            col.update_one({"title": data["title"]}, {"$set": data}, upsert=True)
        except PyMongoError as e:
            st.warning(f"Database sync failed: {e}")

# --- VISUALIZATION ---

def generate_network_graph(main_title: str, references: List[Dict[str, Any]]) -> str:
    """Generates an interactive Pyvis network graph."""
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # Central Node
    net.add_node(main_title, label=main_title[:40], color="#97c2fc", size=30, title=main_title)
    
    # Reference Nodes
    for ref in (references or []):
        ref_title = ref.get('title', 'Unknown Reference')
        net.add_node(ref_title, label=ref_title[:30] + "...", color="#fb7e81", size=15)
        net.add_edge(main_title, ref_title)
        
    return net.generate_html()

# --- USER INTERFACE ---

def main():
    manager = ResearchManager()
    
    st.title("Research Grapher 🚀")
    st.markdown(f"Structure your research citations instantly using **{manager.model_name}**.")
    
    with st.sidebar:
        st.header("Upload Center")
        uploaded_file = st.file_uploader("Choose a Research PDF", type="pdf")
        st.divider()
        st.info("💡 Analysis covers the first 16,000 characters.")

    if not uploaded_file:
        st.info("Please upload a PDF to begin analysis.")
        return

    # Processing Logic
    with st.spinner("Analyzing document structure..."):
        raw_text = manager.parse_pdf(uploaded_file)
        
        if not raw_text:
            st.warning("The PDF appears to be empty or unreadable.")
            return

        # Use cache_data via a wrapper or keep it inside logic if dynamic
        data = manager.extract_metadata(raw_text)

        if "error" in data:
            st.error(data["error"])
            return

        manager.save_paper(data)

    # Layout Results
    col_meta, col_graph = st.columns([1, 2])
    
    with col_meta:
        st.subheader("📄 Paper Metadata")
        st.success(f"**Title:** {data.get('title', 'Unknown')}")
        st.write(f"**Authors:** {', '.join(data.get('authors', ['Not identified']))}")
        
        st.subheader("📚 Key References")
        st.dataframe(data.get('references', []), use_container_width=True)

    with col_graph:
        st.subheader("🕸️ Citation Network")
        html_graph = generate_network_graph(
            data.get('title', 'Target Paper'), 
            data.get('references', [])
        )
        components.html(html_graph, height=600)

if __name__ == "__main__":
    main()
