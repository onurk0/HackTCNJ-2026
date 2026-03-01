import os
import json
import certifi
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
from dataclasses import dataclass, asdict, field

# --- INITIALIZATION ---
load_dotenv()
st.set_page_config(page_title="Research Grapher", page_icon="🚀", layout="wide")

# --- CONSTANTS ---
MODEL_NAME = 'gemini-2.0-flash' # Updated to latest version
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# --- DATA MODELS ---
@dataclass
class PaperMetadata:
    """Encapsulates structured data for a single research paper."""
    title: str
    doi: Optional[str]
    authors: List[str]
    references: List[Dict[str, str]] # [{'title': str, 'doi': str, 'year': str}]
    collection_id: str
    user_id: str 
    # New Fields for Enhanced Tagging
    field_of_study: str = "Unknown"
    perspective: str = "Unknown" # e.g., Theoretical, Applied
    methodologies: List[str] = field(default_factory=list)
    paper_type: str = "Unknown" # e.g., Review, Original Research
    publication_type: str = "Unknown" # e.g., Journal, Conference, Preprint
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
    """Handles all CRUD operations with MongoDB."""
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
        """Persists paper metadata using upsert logic based on unique identifiers."""
        try:
            # Create query based on available unique identifiers
            match_query = {"user_id": paper.user_id, "collection_id": paper.collection_id}
            if paper.doi:
                match_query["doi"] = paper.doi
            else:
                match_query["title"] = paper.title

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

# --- AI PROCESSING LAYER ---
class AIAnalyzer:
    """Handles interaction with Gemini for text analysis and data extraction."""
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Analyzes text to extract structured citation data and research tags."""
        
        # Limit text to context window constraints
        prompt = (
            "Analyze the following research paper text. "
            "Return a JSON object with the following structure:\n"
            "{\n"
            "  'title': '...', 'doi': '...' or null, 'authors': ['...', '...'],\n"
            "  'references': [{'title': '...', 'doi': '...' or null, 'year': '...'}],\n"
            "  'field_of_study': '...', \n"
            "  'perspective': '...', \n"
            "  'methodologies': ['...', '...'], \n"
            "  'paper_type': '...', \n"
            "  'publication_type': '...'\n"
            "}\n"
            "Extract as much detail as possible. Use 'Unknown' if information is not found.\n\n"
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
    """Generates interactive network graphs using Pyvis."""
    @staticmethod
    def generate_comprehensive_graph(papers: List[Dict[str, Any]]) -> str:
        """Generates a graph visualizing relationships between papers and references."""
        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False, cdn_resources='remote')
        
        main_node_color = "#3366cc"
        ref_node_color = "#ff9900"
        
        added_nodes = set()
        
        # Add all papers in the collection as nodes
        for paper in papers:
            paper_id = paper.get('doi') or paper['title']
            
            if paper_id not in added_nodes:
                # Add enhanced metadata to tooltip
                tooltip = f"Field: {paper.get('field_of_study', 'N/A')}\nType: {paper.get('paper_type', 'N/A')}"
                net.add_node(paper_id, label=paper['title'][:20] + "...", color=main_node_color, title=tooltip, shape="box")
                added_nodes.add(paper_id)
            
            references = paper.get('references', [])
            
            for ref in references:
                ref_id = ref.get('doi') or ref.get('title')
                if not ref_id: continue
                
                if ref_id not in added_nodes:
                    # Check if this ref exists in our collection
                    is_in_collection = any(p.get('doi') == ref_id for p in papers)
                    
                    if is_in_collection:
                        net.add_node(ref_id, label=ref.get('title')[:20] + "...", color=main_node_color, shape="box")
                    else:
                        net.add_node(ref_id, label=ref.get('title')[:20] + "...", color=ref_node_color)
                    
                    added_nodes.add(ref_id)
                
                net.add_edge(paper_id, ref_id) 
        
        net.set_options("""
        var options = {
          "nodes": {
            "font": { "size": 12 }
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -10000,
              "springLength": 200
            },
            "minVelocity": 0.75
          }
        }
        """)
        
        return net.generate_html()

# --- CORE LOGIC ---
def parse_pdf(file) -> str:
    """Extracts text from an uploaded PDF file securely."""
    try:
        reader = PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text.strip()
    except Exception as e:
        st.error(f"Failed to read PDF: {e}")
        return ""

def save_file_to_server(uploaded_file) -> str:
    """Saves file locally and returns the path."""
    os.makedirs("storage", exist_ok=True)
    file_path = f"storage/{uuid.uuid4()}_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# --- USER INTERFACE ---

def render_results(papers: List[Dict[str, Any]], collection_id: str):
    """Displays data tables and graphs for the active collection."""
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📄 Collection Data")
        
        # Display extracted tags in a table
        for paper in papers:
            with st.expander(f"{paper['title'][:30]}..."):
                st.write(f"**Field:** {paper.get('field_of_study', 'Unknown')}")
                st.write(f"**Methodologies:** {', '.join(paper.get('methodologies', []))}")
                st.write(f"**Type:** {paper.get('paper_type', 'Unknown')} ({paper.get('publication_type', 'Unknown')})")
                st.write(f"**Perspective:** {paper.get('perspective', 'Unknown')}")
        
    with col2:
        st.subheader("🕸️ Comprehensive Citation Network")                
        
        graph_html = GraphVisualizer.generate_comprehensive_graph(papers)
        components.html(graph_html, height=720)

def render_ui():
    """Main UI rendering logic with workspace management."""
    st.title("Research Grapher 🚀")
    st.markdown(f"Structure and analyze research citations using **{MODEL_NAME}**.")
    
    # Mock user session
    user_id = "user_123"

    # Initialize Managers
    db_manager = PaperDataManager(MONGO_URI)
    ai_analyzer = AIAnalyzer(GEMINI_API_KEY)

    # --- WORKSPACE MANAGEMENT ---
    with st.sidebar:
        st.header("Workspace")
        existing_collections = db_manager.get_collections(user_id)
        
        new_collection_name = st.text_input("Create New Collection")
        collection_option = st.selectbox(
            "Select Existing",
            options=["-- Select --"] + existing_collections
        )
        
        if new_collection_name:
            active_collection = new_collection_name
        elif collection_option != "-- Select --":
            active_collection = collection_option
        else:
            active_collection = None

        st.divider()
        st.info("Upload PDFs to map the collection.")
    
    # --- MAIN CONTENT AREA ---
    if active_collection:
        st.subheader(f"Active Workspace: **{active_collection}**")
        
        uploaded_files = st.file_uploader(
            "Upload Research PDFs", 
            type="pdf", 
            accept_multiple_files=True 
        )

        if uploaded_files:
            with st.spinner("🧠 AI is analyzing papers and extracting tags..."):
                for uploaded_file in uploaded_files:
                    # 1. Store file
                    file_path = save_file_to_server(uploaded_file)
                    # 2. Extract text
                    raw_text = parse_pdf(uploaded_file)
                    # 3. Analyze text via AI
                    extracted_data = ai_analyzer.extract_metadata(raw_text)

                    if "error" not in extracted_data:
                        # 4. Create structured object
                        current_paper = PaperMetadata(
                            title=extracted_data.get('title', 'Unknown'),
                            doi=extracted_data.get('doi'),
                            authors=extracted_data.get('authors', []),
                            references=extracted_data.get('references', []),
                            collection_id=active_collection, 
                            user_id=user_id,
                            field_of_study=extracted_data.get('field_of_study', 'Unknown'),
                            perspective=extracted_data.get('perspective', 'Unknown'),
                            methodologies=extracted_data.get('methodologies', []),
                            paper_type=extracted_data.get('paper_type', 'Unknown'),
                            publication_type=extracted_data.get('publication_type', 'Unknown'),
                            file_path=file_path
                        )
                        # 5. Save to DB
                        db_manager.save_paper(current_paper)
                    else:
                        st.error(extracted_data["error"])
            
            st.success("Files processed successfully!")
        
        # Load and render data
        saved_papers = db_manager.get_papers_by_collection(user_id, active_collection)
        if saved_papers:
            render_results(saved_papers, active_collection)
        else:
            st.info("No papers in this collection yet.")

    else:
        st.info("Select or create a collection in the sidebar.")

if __name__ == "__main__":
    render_ui()
