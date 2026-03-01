import os
import json
import certifi
import ssl
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pyvis.network import Network
from typing import Dict, Any, List, Optional
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dataclasses import dataclass

# --- INITIALIZATION ---
load_dotenv()
st.set_page_config(page_title="Research Grapher", page_icon="🚀", layout="wide")

# --- CONSTANTS ---
MODEL_NAME = 'gemini-2.5-flash'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# --- DATA MODELS ---
@dataclass
class PaperMetadata:
    title: str
    authors: List[str]
    references: List[Dict[str, str]] # [{'title': str, 'year': str}]
    collection_id: str
    raw_text: Optional[str] = None


# --- RESOURCE MANAGEMENT ---
@st.cache_resource
def get_ai_client() -> genai.Client:
    """Initializes the Google GenAI Client as a cached resource."""
    if not GEMINI_API_KEY:
        st.error("Missing GEMINI_API_KEY in environment.")
        st.stop()
    return genai.Client(api_key=GEMINI_API_KEY)


# --- DATABASE LAYER ---
class PaperDataManager:
    def __init__(self, uri: str):
        self.uri = uri
        self.collection = self._get_collection()

    @st.cache_resource
    def _get_collection(_self):
        """Establishes MongoDB connection safely."""
        try:
            # Use standard SSL settings for Atlas
            client = MongoClient(_self.uri, tlsCAFile=certifi.where())
            client.admin.command('ping')
            return client["research_db"]["papers"]
        except PyMongoError as e:
            st.error(f"Database connection failed: {e}")
            st.stop()

    def get_papers_by_collection(self, collection_id: str):
        """Fetches all papers belonging to a specific collection."""
        return list(self.collection.find({"collection_id": collection_id}))

    def save_paper(self, paper: PaperMetadata):
        """Saves paper with a collection_id."""
        self.collection.update_one(
            {"title": paper.title, "collection_id": paper.collection_id},
            {"$set": {
                "title": paper.title,
                "authors": paper.authors,
                "references": paper.references,
                "collection_id": paper.collection_id
            }},
            upsert=True
        )

    def save_paper(self, paper: PaperMetadata):
        """Persists paper metadata using upsert logic."""
        try:
            self.collection.update_one(
                {"title": paper.title},
                {"$set": {
                    "title": paper.title,
                    "authors": paper.authors,
                    "references": paper.references
                }},
                upsert=True
            )
        except PyMongoError as e:
            st.error(f"Failed to save to database: {e}")


# --- AI PROCESSING LAYER ---
class AIAnalyzer:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Analyzes text to extract structured citation data."""
        
        # Limit text to context window constraints
        prompt = (
            "Analyze the following research paper text. "
            "Return a JSON object with keys: 'title' (string), 'authors' (list of strings), "
            "and 'references' (list of objects with 'title' and 'year'). "
            "Focus on extracting the main title and the bibliography section.\n\n"
            f"Text snippet: {text[:20000]}"
        )

        try:
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={'response_mime_type': 'application/json'}
            )
            return json.loads(response.text)
        except Exception as e:
            return {"error": f"AI Processing failed: {str(e)}"}

# --- VISUALIZATION LAYER ---
class GraphVisualizer:
    @staticmethod
    def generate_comprehensive_graph(papers: List[PaperMetadata]) -> str:
        """Generates graph based on all papers in a collection."""
        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False, cdn_resources='remote')
        
        main_node_color = "#3366cc"
        ref_node_color = "#ff9900"
        
        # Add all papers in the collection as main nodes
        for paper in papers:
            net.add_node(paper.title, color=main_node_color, title="Main Paper", shape="box")
            
            # 2. Add references as nodes and link them
            for ref in paper.references:
                ref_title = ref.get('title')
                if ref_title:
                    net.add_node(ref_title, color=ref_node_color, title="Reference", size = 10)
                    net.add_edge(paper.title, ref_title, color = "#cccccc")
        net.set_options("""
        var options = {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "LR",
              "sortMethod": "directed"
            }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -10000,
              "centralGravity": 0.3,
              "springLength": 250,
              "springConstant": 0.01
            },
            "minVelocity": 0.75
          }
        }
        """)
        
        return net.generate_html()

    @staticmethod
    def generate_litmap_style(main_paper: PaperMetadata) -> str:
        """Generates a professional, interactive citation network graph."""
        
        # Initialize Network with hierarchical layout capabilities
        net = Network(
            height="700px", 
            width="100%", 
            bgcolor="#ffffff", 
            font_color="black",
            notebook=False,
            cdn_resources='remote'
        )
        
        # Styling nodes
        main_node_color = "#3366cc"
        ref_node_color = "#ff9900"
        
        # Add central node (The uploaded paper)
        net.add_node(
            main_paper.title, 
            label=main_paper.title[:40] + "..." if len(main_paper.title) > 40 else main_paper.title, 
            color=main_node_color, 
            size=30, 
            title=f"<b>Title:</b> {main_paper.title}<br><b>Authors:</b> {', '.join(main_paper.authors)}",
            shape="box"
        )
        
        # Add and link reference nodes
        for ref in main_paper.references:
            ref_title = ref.get('title', 'Unknown')
            ref_year = ref.get('year', 'N/A')
            
            node_id = f"{ref_title} ({ref_year})"
            
            net.add_node(
                node_id, 
                label=ref_title[:25] + "...", 
                color=ref_node_color, 
                size=15,
                title=f"<b>Title:</b> {ref_title}<br><b>Year:</b> {ref_year}"
            )
            net.add_edge(main_paper.title, node_id, color="#999999", weight=1)

        # Advanced physics configuration for better layout
        net.set_options("""
        var options = {
          "nodes": {
            "font": { "size": 12 }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -8000,
              "centralGravity": 0.3,
              "springLength": 150,
              "springConstant": 0.04
            },
            "minVelocity": 0.75
          }
        }
        """)
        
        return net.generate_html()

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
   
    # Initialize Core Classes
    db_manager = PaperDataManager(MONGO_URI)
    ai_analyzer = AIAnalyzer(GEMINI_API_KEY)

    with st.sidebar:
        st.header("1. Collection Name")
        collection_id = st.text_input("Enter collection name", value="My Research")
        st.divider()
        st.info("💡 Upload multiple PDFs to create a comprehensive graph.")
        
    st.subheader("Upload Papers")
    uploaded_files = st.file_uploader(
        "Upload Research PDFs", 
        type="pdf", 
        accept_multiple_files=True 
    )

    if uploaded_files:
        st.divider()
        # Create a container for results
        results_container = st.container()
        
        with results_container:
            with st.spinner("🧠 AI is reading all papers..."):
                all_papers_data = []
                
                for uploaded_file in uploaded_files:
                    raw_text = parse_pdf(uploaded_file)
                    if not raw_text:
                        continue

                    # Extract Metadata
                    extracted_data = ai_analyzer.extract_metadata(raw_text)
                    
                    if "error" in extracted_data:
                        st.error(f"Error processing {uploaded_file.name}: {extracted_data['error']}")
                        continue
                    
                    # Create Data Object (including Collection ID)
                    current_paper = PaperMetadata(
                        title=extracted_data.get('title', 'Unknown'),
                        authors=extracted_data.get('authors', []),
                        references=extracted_data.get('references', []),
                        collection_id=collection_id, # Link to collection
                        raw_text=raw_text
                    )
                    
                    # Save to Database
                    db_manager.save_paper(current_paper)
                    all_papers_data.append(current_paper)

            if all_papers_data:
                # Layout Results
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("📄 Collection Summary")
                    st.write(f"**Collection:** {collection_id}")
                    st.write(f"**Papers Analyzed:** {len(all_papers_data)}")
                    
                    # Display papers in a table
                    paper_titles = [p.title for p in all_papers_data]
                    st.dataframe(paper_titles, width='stretch')
                    
                with col2:
                    st.subheader("🕸️ Comprehensive Citation Network")
                    # Generate graph using ALL papers in the collection
                    graph_html = GraphVisualizer.generate_comprehensive_graph(all_papers_data)
                    components.html(graph_html, height=720)

if __name__ == "__main__":
    render_ui()
