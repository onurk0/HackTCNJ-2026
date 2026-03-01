import os
import json
import certifi
import ssl
import uuid
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pyvis.network import Network
from typing import Dict, Any, List, Optional
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from dataclasses import dataclass, asdict

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
    doi: Optional[str]
    authors: List[str]
    references: List[Dict[str, str]] # [{'title': str, 'year': str}]
    collection_id: str
    user_id: str 
    raw_text: Optional[str] = None
    file_path: Optional[str] = None


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
        self.client = None
        self.db = None
        self._connect()

    def _connect(self):
        """Establishes MongoDB connection securely."""
        try:
            self.client = MongoClient(self.uri, tlsCAFile=certifi.where())
            # Ping to confirm connection
            self.client.admin.command('ping')
            self.db = self.client["research_db"]
        except PyMongoError as e:
            st.error(f"Database connection failed: {e}")
            st.stop()

    def get_collections(self, user_id: str) -> List[str]:
        """Retrieves distinct collection names for a specific user."""
        try:
            papers = self.db["papers"].find({"user_id": user_id})
            # Extract unique collection names
            return list(set(p['collection_id'] for p in papers if 'collection_id' in p))
        except PyMongoError as e:
            st.error(f"Failed to fetch collections: {e}")
            return []

    def save_paper(self, paper: PaperMetadata):
        """Persists paper metadata using upsert logic based on title, user, and collection."""
        try:
            match_query = {"user_id": paper.user_id, "collection_id": paper.collection_id}
            if paper.doi:
                match_query["doi"] = paper.doi # Prioritize DOI
            else:
                match_query["title"] = paper.title # Fallback to Title

            self.db["papers"].update_one(
                match_query,
                {"$set": asdict(paper)},
                upsert=True
            )
        except PyMongoError as e:
            st.error(f"Failed to save to database: {e}")
            
    def get_papers_by_collection(self, user_id: str, collection_id: str) -> List[Dict[str, Any]]:
        """Retrieves all papers for a specific user and collection."""
        try:
            return list(self.db["papers"].find({"user_id": user_id, "collection_id": collection_id}))
        except PyMongoError as e:
            st.error(f"Failed to query database: {e}")
            return []
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
            "Return a JSON object with keys: 'title' (string), 'doi' (string or null), "
            "'authors' (list of strings), and 'references' (list of objects with 'title', "
            "'doi' (string or null), and 'year'). "
            "Extract the DOI for the main paper if available, and try to find DOIs for references.\n\n"
            f"Text: {text[:16000]}"
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
    def generate_comprehensive_graph(papers: List[Dict[str, Any]]) -> str:
        """Generates graph based on all papers in a collection."""
        # Use notebook=False for Streamlit
        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False, cdn_resources='remote')
        
        main_node_color = "#3366cc"
        ref_node_color = "#ff9900"
        
        # Track added nodes to prevent duplicate ID errors
        added_nodes = set()
        
        # Add all papers in the collection as main nodes
        for paper in papers:
            # Uses DOIs as unique identifiers
            paper_id = paper.get('doi') or paper['title']

            if paper_id not in added_nodes:
                net.add_node(paper_id, label=paper['title'][:30] + "...", color=main_node_color, title=paper['title'], shape="box")
                added_nodes.add(paper_id)
            
            references = paper.get('references', [])
            
            for ref in references:
                # If references are stored as stringified JSON in DB, 
                # we might need to parse them, but usually Mongo handles this.
                ref_id = ref.get('doi') or ref.get('title')
                if not ref_id: continue
                
                if ref_id not in added_nodes:
                    # Check if this ref exists in our collection via DOI
                    is_in_collection = any(p.get('doi') == ref_id for p in papers)
                    
                    if is_in_collection:
                        net.add_node(ref_id, label=ref.get('title')[:30] + "...", color=main_node_color, shape="box")
                    else:
                        net.add_node(ref_id, label=ref.get('title')[:30] + "...", color=ref_node_color, title=ref.get('title'))
                    
                    added_nodes.add(ref_id)
                
                net.add_edge(paper_id, ref_id) 
        
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
            doi=extracted_data.get('doi'),
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

def save_file_to_server(uploaded_file) -> str:
    """Saves file locally and returns the path."""
    # Create a directory named 'storage' if it doesn't exist
    os.makedirs("storage", exist_ok=True)
    # Create a unique filename
    file_path = f"storage/{uuid.uuid4()}_{uploaded_file.name}"
    # Save the bytes
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

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

def render_results(papers: List[Dict[str, Any]], collection_id: str):
    """Displays dataframes and graphs for pre-loaded papers."""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📄 Collection Summary")
        st.write(f"**Collection:** {collection_id}")
        st.write(f"**Papers:** {len(papers)}")
        
        # Extract titles from the list of dictionary documents
        paper_titles = [p['title'] for p in papers]
        st.dataframe(paper_titles, width='stretch')
        
    with col2:
        st.subheader("🕸️ Comprehensive Citation Network")                
        
        graph_html = GraphVisualizer.generate_comprehensive_graph(papers)
        components.html(graph_html, height=720)

def render_ui():
    """Main UI rendering logic with collection management."""
    st.title("Research Grapher 🚀")
    st.markdown(f"Structure your research citations instantly using **{MODEL_NAME}**.")
    
    # --- AUTHENTICATION MOCKUP ---
    # In multi-user app, this comes from login session (e.g., st.session_state)
    user_id = "user_123"

    # Initialize Core Classes
    db_manager = PaperDataManager(MONGO_URI)
    ai_analyzer = AIAnalyzer(GEMINI_API_KEY)

    # --- COLLECTION MANAGEMENT ---
    with st.sidebar:
        st.header("Workspace")
        
        # Fetch existing collections
        existing_collections = db_manager.get_collections(user_id)
        
        # Option to create a new collection
        new_collection_name = st.text_input("Or create new collection")
        
        # Selector for existing or new
        collection_option = st.selectbox(
            "Select Collection",
            options=["-- Select --"] + existing_collections
        )
        
        # Determine active collection
        if new_collection_name:
            active_collection = new_collection_name
        elif collection_option != "-- Select --":
            active_collection = collection_option
        else:
            active_collection = None

        st.divider()
        st.info("💡 Upload PDFs to create a graph for the active collection.")
    
    # --- MAIN CONTENT AREA ---
    if active_collection:
        st.subheader(f"Current Workspace: **{active_collection}**")
        
        saved_papers = db_manager.get_papers_by_collection(user_id, active_collection)

        st.divider()
        st.subheader("Add Papers to Collection")
        uploaded_files = st.file_uploader(
            "Upload Research PDFs", 
            type="pdf", 
            accept_multiple_files=True 
        )

        if uploaded_files:
            with st.spinner("🧠 AI is reading new papers..."):
                for uploaded_file in uploaded_files:
                    file_path = save_file_to_server(uploaded_file)
                    raw_text = parse_pdf(uploaded_file)
                    extracted_data = ai_analyzer.extract_metadata(raw_text)

                    # Create & Save object
                    current_paper = PaperMetadata(
                        title=extracted_data.get('title', 'Unknown'),
                        doi=extracted_data.get('doi'),
                        authors=extracted_data.get('authors', []),
                        references=extracted_data.get('references', []),
                        collection_id=active_collection, 
                        user_id=user_id,
                        file_path=file_path
                    )
                    db_manager.save_paper(current_paper)
            
            saved_papers = db_manager.get_papers_by_collection(user_id, active_collection)

        if saved_papers:
            st.write(f"Found {len(saved_papers)} papers in collection.")
            render_results(saved_papers, active_collection)
        else:
            st.info("No papers saved for this collection yet.")

    else:
        st.info("Please select or create a collection in the sidebar to begin.")

if __name__ == "__main__":
    render_ui()
