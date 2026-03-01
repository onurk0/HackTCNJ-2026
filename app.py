"""
Research Grapher – Main Application
=====================================
A Streamlit-based tool that uses Google Gemini to extract metadata from
research PDFs, persists results to MongoDB, and renders interactive citation
networks with **connection-strength visualizations**.

Connection strength is computed by counting how many papers in the active
collection mutually reference each other, then encoded as:
  • Edge width   – thicker = stronger link
  • Edge color   – gradient from light-grey (#cccccc) → vivid orange (#ff6600)
  • Node size    – larger = more citations received from sibling papers
  • A Streamlit sidebar legend explains every visual encoding to the user

Architecture
------------
  PaperMetadata       – immutable dataclass representing one paper
  PaperDataManager    – all MongoDB I/O (database layer)
  AIAnalyzer          – Google Gemini prompts (AI layer)
  ConnectionStrength  – pure functions that score inter-paper links
  GraphVisualizer     – Pyvis graph builders (visualization layer)
  render_*            – thin Streamlit UI functions (UI layer)
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import os
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import certifi
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from pyvis.network import Network
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
st.set_page_config(page_title="Research Grapher", page_icon="🚀", layout="wide")

# ---------------------------------------------------------------------------
# Global constants
# ---------------------------------------------------------------------------
MODEL_NAME: str = "gemini-2.5-flash"
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
MONGO_URI: str = os.getenv("MONGO_URI", "")

# Visual encoding parameters for connection-strength graph
_EDGE_MIN_WIDTH: float = 1.0       # px – weakest link
_EDGE_MAX_WIDTH: float = 10.0      # px – strongest link
_NODE_BASE_SIZE: int = 20          # base size for main-collection nodes
_NODE_SIZE_PER_CITATION: int = 8   # extra px per inbound citation
_COLOR_WEAK_EDGE: str = "#cccccc"  # hex – low-strength edge
_COLOR_STRONG_EDGE: str = "#ff6600"  # hex – high-strength edge


# ===========================================================================
# DATA MODEL
# ===========================================================================

@dataclass
class PaperMetadata:
    """
    Represents the structured metadata extracted from one research paper.

    Attributes
    ----------
    title            : Canonical paper title.
    doi              : Digital Object Identifier (may be None).
    authors          : Ordered list of author name strings.
    references       : List of {'title', 'doi'?, 'year'?} dicts.
    collection_id    : User-defined workspace name.
    user_id          : Identifier for the owning user.
    raw_text         : First N chars of extracted PDF text (optional).
    file_path        : Server-side storage path (optional).
    field_of_study   : Broad academic discipline.
    perspective      : e.g. "Theoretical", "Applied".
    methodologies    : Techniques used (e.g. ["RCT", "Meta-analysis"]).
    paper_type       : e.g. "Review", "Original Research".
    publication_type : e.g. "Journal", "Conference", "Preprint".
    """

    title: str
    doi: Optional[str]
    authors: List[str]
    references: List[Dict[str, str]]
    collection_id: str
    user_id: str
    raw_text: Optional[str] = None
    file_path: Optional[str] = None
    field_of_study: str = "Unknown"
    perspective: str = "Unknown"
    methodologies: List[str] = field(default_factory=list)
    paper_type: str = "Unknown"
    publication_type: str = "Unknown"


# ===========================================================================
# DATABASE LAYER
# ===========================================================================

class PaperDataManager:
    """
    Encapsulates all interactions with the MongoDB research_db database.

    Responsibilities
    ----------------
    - Establish and verify a TLS-secured connection on construction.
    - Provide collection-scoped CRUD helpers.
    - Surface errors via Streamlit alerts rather than raising raw exceptions
      so the UI layer stays clean.
    """

    def __init__(self, uri: str) -> None:
        """
        Parameters
        ----------
        uri : MongoDB connection string (loaded from .env).
        """
        self.uri = uri
        self.client: Optional[MongoClient] = None
        self.db = None
        self._connect()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """Opens a TLS-secured MongoDB connection and pings to confirm."""
        try:
            self.client = MongoClient(self.uri, tlsCAFile=certifi.where())
            self.client.admin.command("ping")       # fast liveness check
            self.db = self.client["research_db"]
        except PyMongoError as exc:
            st.error(f"Database connection failed: {exc}")
            st.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_collections(self, user_id: str) -> List[str]:
        """
        Returns all distinct collection names belonging to *user_id*.

        Parameters
        ----------
        user_id : The authenticated user's identifier.
        """
        try:
            docs = self.db["papers"].find({"user_id": user_id})
            return list({p["collection_id"] for p in docs if "collection_id" in p})
        except PyMongoError as exc:
            st.error(f"Failed to fetch collections: {exc}")
            return []

    def save_paper(self, paper: PaperMetadata) -> None:
        """
        Upserts paper metadata into the 'papers' collection.

        Matching priority: DOI (preferred) > title + user + collection.

        Parameters
        ----------
        paper : Fully populated PaperMetadata instance.
        """
        try:
            # Build the uniqueness filter for upsert
            match_query: Dict[str, Any] = {
                "user_id": paper.user_id,
                "collection_id": paper.collection_id,
            }
            if paper.doi:
                match_query["doi"] = paper.doi      # DOI is globally unique
            else:
                match_query["title"] = paper.title  # fall back to title

            self.db["papers"].update_one(
                match_query,
                {"$set": asdict(paper)},
                upsert=True,
            )
        except PyMongoError as exc:
            st.error(f"Failed to save to database: {exc}")

    def get_papers_by_collection(
        self, user_id: str, collection_id: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all papers for a specific user/collection pair.

        Parameters
        ----------
        user_id       : Owning user.
        collection_id : Target collection name.

        Returns
        -------
        List of raw MongoDB document dicts.
        """
        try:
            return list(
                self.db["papers"].find(
                    {"user_id": user_id, "collection_id": collection_id}
                )
            )
        except PyMongoError as exc:
            st.error(f"Failed to query database: {exc}")
            return []


# ===========================================================================
# AI LAYER
# ===========================================================================

class AIAnalyzer:
    """
    Wraps Google Gemini to extract structured metadata from raw paper text.

    The class is intentionally slim: one public method, one prompt template.
    Swap the underlying model or provider here without touching callers.
    """

    def __init__(self, api_key: str) -> None:
        """
        Parameters
        ----------
        api_key : Gemini API key string.
        """
        self.client = genai.Client(api_key=api_key)

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Sends a JSON-mode prompt to Gemini and returns parsed metadata.

        Parameters
        ----------
        text : Raw text extracted from a PDF (may be very long).

        Returns
        -------
        Dict with keys: title, doi, authors, references, field_of_study,
        perspective, methodologies, paper_type, publication_type.
        On failure, returns {'error': <message>}.
        """
        prompt = (
            "Analyze the following research paper text. "
            "Return a JSON object with the following structure:\n"
            "{\n"
            "  'title': '...', 'doi': '...' or null, 'authors': ['...'],\n"
            "  'references': [{'title': '...', 'doi': '...' or null, 'year': '...'}],\n"
            "  'field_of_study': '...', 'perspective': '...',\n"
            "  'methodologies': ['...'], 'paper_type': '...',\n"
            "  'publication_type': '...'\n"
            "}\n"
            "Use 'Unknown' for missing fields. Extract as much detail as possible.\n\n"
            f"Text snippet: {text[:20000]}"
        )

        try:
            response = self.client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            return json.loads(response.text)
        except Exception as exc:  # broad catch – Gemini SDK raises many types
            return {"error": f"AI Processing failed: {str(exc)}"}


# ===========================================================================
# CONNECTION-STRENGTH LAYER
# ===========================================================================

class ConnectionStrength:
    """
    Pure-function utilities to score how strongly papers are connected.

    Connection strength between paper A and paper B increases by 1 for each
    of the following that is true:
      1. A cites B  (A's reference list contains B's DOI or title)
      2. B cites A  (B's reference list contains A's DOI or title)

    Final score therefore ranges [0, 2] per pair, then sums across the
    entire collection to derive per-node citation counts.

    No I/O. All methods are static for easy unit testing.
    """

    @staticmethod
    def _get_paper_id(paper: Dict[str, Any]) -> str:
        """Returns a stable identifier: DOI if available, else title."""
        return paper.get("doi") or paper.get("title", "")

    @staticmethod
    def _get_ref_id(ref: Dict[str, str]) -> str:
        """Returns the best available identifier for a reference entry."""
        return ref.get("doi") or ref.get("title", "")

    @staticmethod
    def _paper_cites(paper: Dict[str, Any], target_id: str) -> bool:
        """
        Returns True if *paper* has a reference whose id matches *target_id*.

        Parameters
        ----------
        paper     : MongoDB document dict with a 'references' list.
        target_id : DOI or title string to look for.
        """
        if not target_id:
            return False
        for ref in paper.get("references", []):
            if ConnectionStrength._get_ref_id(ref) == target_id:
                return True
        return False

    @staticmethod
    def compute_edge_weights(
        papers: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], int]:
        """
        Computes pairwise connection strengths for all papers in a collection.

        Each ordered pair (src_id, ref_id) from src's reference list gets +1.
        If that reference also exists as a main paper that back-cites src,
        the undirected edge weight between the two is the sum of both directions.

        Parameters
        ----------
        papers : All paper documents for the collection.

        Returns
        -------
        Dict mapping frozenset-style sorted tuple (id_a, id_b) → int strength.
        """
        weight: Dict[Tuple[str, str], int] = {}

        for i, paper_a in enumerate(papers):
            id_a = ConnectionStrength._get_paper_id(paper_a)

            for paper_b in papers[i + 1 :]:   # only upper triangle
                id_b = ConnectionStrength._get_paper_id(paper_b)

                score = 0
                if ConnectionStrength._paper_cites(paper_a, id_b):
                    score += 1
                if ConnectionStrength._paper_cites(paper_b, id_a):
                    score += 1

                if score > 0:
                    key = (id_a, id_b) if id_a < id_b else (id_b, id_a)
                    weight[key] = weight.get(key, 0) + score

        return weight

    @staticmethod
    def compute_inbound_citation_counts(
        papers: List[Dict[str, Any]],
    ) -> Dict[str, int]:
        """
        Counts how many sibling papers in the collection cite each paper.

        Used to size main-collection nodes proportional to their influence.

        Parameters
        ----------
        papers : All paper documents for the collection.

        Returns
        -------
        Dict mapping paper_id → inbound citation count (within collection).
        """
        counts: Dict[str, int] = {}
        ids = {ConnectionStrength._get_paper_id(p) for p in papers}

        for paper in papers:
            for ref in paper.get("references", []):
                ref_id = ConnectionStrength._get_ref_id(ref)
                if ref_id in ids:
                    counts[ref_id] = counts.get(ref_id, 0) + 1

        return counts

    @staticmethod
    def interpolate_color(strength: int, max_strength: int) -> str:
        """
        Returns a hex color string interpolated between weak and strong colors.

        Weak  → _COLOR_WEAK_EDGE   (#cccccc, light grey)
        Strong → _COLOR_STRONG_EDGE (#ff6600, vivid orange)

        Parameters
        ----------
        strength     : Observed edge weight (1 or 2 for pairwise).
        max_strength : Maximum weight in the collection (for normalisation).
        """
        if max_strength == 0:
            return _COLOR_WEAK_EDGE

        # Normalise to [0, 1]
        t = min(strength / max_strength, 1.0)

        # Parse component channels from global hex constants
        def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
            h = h.lstrip("#")
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

        r1, g1, b1 = _hex_to_rgb(_COLOR_WEAK_EDGE)
        r2, g2, b2 = _hex_to_rgb(_COLOR_STRONG_EDGE)

        # Linear interpolation per channel
        r = int(r1 + t * (r2 - r1))
        g = int(g1 + t * (g2 - g1))
        b = int(b1 + t * (b2 - b1))

        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def scale_edge_width(strength: int, max_strength: int) -> float:
        """
        Maps edge strength to a pixel width in [_EDGE_MIN_WIDTH, _EDGE_MAX_WIDTH].

        Parameters
        ----------
        strength     : Edge weight.
        max_strength : Maximum weight in the graph.
        """
        if max_strength == 0:
            return _EDGE_MIN_WIDTH
        t = min(strength / max_strength, 1.0)
        return _EDGE_MIN_WIDTH + t * (_EDGE_MAX_WIDTH - _EDGE_MIN_WIDTH)


# ===========================================================================
# VISUALIZATION LAYER
# ===========================================================================

class GraphVisualizer:
    """
    Builds Pyvis-based interactive HTML graphs.

    Two public factory methods are provided:
      - generate_comprehensive_graph : Full collection view with
            connection-strength encoding on edges and nodes.
      - generate_litmap_style        : Single-paper star/ego graph.

    Both methods return a raw HTML string suitable for
    ``streamlit.components.v1.html()``.
    """

    # Color palette – centralised so UI legend stays in sync
    COLOR_MAIN_NODE: str = "#3366cc"    # blue  – paper is in the collection
    COLOR_REF_NODE: str = "#ff9900"     # amber – external reference only
    COLOR_MAIN_PAPER: str = "#22aa44"   # green – focal paper in litmap view

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _base_network() -> Network:
        """Returns a pre-configured Pyvis Network object."""
        return Network(
            height="720px",
            width="100%",
            bgcolor="#f8f9fa",
            font_color="#111111",
            notebook=False,
            cdn_resources="remote",
        )

    @staticmethod
    def _apply_physics(net: Network) -> None:
        """Applies Barnes-Hut physics options for stable, readable layouts."""
        net.set_options(
            """
            var options = {
              "nodes": { "font": { "size": 13, "face": "Arial" } },
              "edges": { "smooth": { "type": "dynamic" } },
              "physics": {
                "barnesHut": {
                  "gravitationalConstant": -9000,
                  "centralGravity": 0.25,
                  "springLength": 160,
                  "springConstant": 0.04,
                  "damping": 0.09
                },
                "minVelocity": 0.6,
                "stabilization": { "iterations": 150 }
              }
            }
            """
        )

    @staticmethod
    def _truncate(label: str, max_len: int = 30) -> str:
        """Truncates a string and appends ellipsis if over max_len."""
        return label[:max_len] + "…" if len(label) > max_len else label

    # ------------------------------------------------------------------
    # Public factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def generate_comprehensive_graph(papers: List[Dict[str, Any]]) -> str:
        """
        Builds a full collection citation graph with connection-strength encoding.

        Visual encodings
        ----------------
        Node size (main papers)  : Larger = more citations from sibling papers
        Edge width               : Thicker = more mutual citations (1–10 px)
        Edge color               : Grey → orange gradient (weak → strong)
        Edge tooltip             : Shows numeric strength score

        Parameters
        ----------
        papers : All paper documents retrieved from MongoDB for the collection.

        Returns
        -------
        HTML string for embedding via streamlit.components.v1.html().
        """
        net = GraphVisualizer._base_network()

        # ---- Pre-compute connection metrics ----
        edge_weights = ConnectionStrength.compute_edge_weights(papers)
        inbound_counts = ConnectionStrength.compute_inbound_citation_counts(papers)
        max_weight = max(edge_weights.values(), default=1)

        # Build a fast DOI → paper lookup for membership checks
        collection_ids: Dict[str, Dict] = {
            (p.get("doi") or p["title"]): p for p in papers
        }

        added_nodes: set = set()

        # ---- Add main-collection nodes ----
        for paper in papers:
            paper_id = paper.get("doi") or paper["title"]

            if paper_id not in added_nodes:
                # Scale node size by inbound citations from siblings
                inbound = inbound_counts.get(paper_id, 0)
                node_size = _NODE_BASE_SIZE + inbound * _NODE_SIZE_PER_CITATION

                tooltip = (
                    f"<b>{paper['title']}</b><br>"
                    f"Field: {paper.get('field_of_study', 'Unknown')}<br>"
                    f"Cited by {inbound} sibling paper(s)"
                )

                net.add_node(
                    paper_id,
                    label=GraphVisualizer._truncate(paper["title"]),
                    color=GraphVisualizer.COLOR_MAIN_NODE,
                    size=node_size,
                    title=tooltip,
                    shape="box",
                    borderWidth=2,
                )
                added_nodes.add(paper_id)

        # ---- Add reference nodes and edges ----
        for paper in papers:
            src_id = paper.get("doi") or paper["title"]

            for ref in paper.get("references", []):
                ref_id = ref.get("doi") or ref.get("title")
                if not ref_id:
                    continue  # skip malformed reference entries

                # Determine if this reference is a sibling in the collection
                is_sibling = ref_id in collection_ids

                # Add the reference node only once
                if ref_id not in added_nodes:
                    if is_sibling:
                        # Already handled in the main-node loop above
                        pass
                    else:
                        # External reference node (amber, smaller, ellipse)
                        ref_title = ref.get("title", ref_id)
                        net.add_node(
                            ref_id,
                            label=GraphVisualizer._truncate(ref_title, 25),
                            color=GraphVisualizer.COLOR_REF_NODE,
                            size=12,
                            title=f"<b>{ref_title}</b><br>External reference",
                            shape="ellipse",
                        )
                    added_nodes.add(ref_id)

                # ---- Add edge with strength encoding ----
                # Build canonical (sorted) key to look up pre-computed weight
                key = (src_id, ref_id) if src_id < ref_id else (ref_id, src_id)
                strength = edge_weights.get(key, 0)

                edge_color = ConnectionStrength.interpolate_color(strength, max_weight)
                edge_width = ConnectionStrength.scale_edge_width(strength, max_weight)
                edge_label = f"strength: {strength}" if strength > 0 else ""
                edge_tooltip = (
                    f"Connection strength: {strength}/{max_weight}<br>"
                    f"{'Mutual citation' if strength == 2 else 'One-way citation'}"
                )

                net.add_edge(
                    src_id,
                    ref_id,
                    color=edge_color,
                    width=edge_width,
                    title=edge_tooltip,
                    label=edge_label if is_sibling and strength > 0 else "",
                    arrows="to",
                )

        GraphVisualizer._apply_physics(net)
        return net.generate_html()

    @staticmethod
    def generate_litmap_style(main_paper: PaperMetadata) -> str:
        """
        Generates a star/ego graph centred on a single uploaded paper.

        The focal paper is shown in green; its references radiate outward
        in amber.  Connection strength is not applicable here (single paper),
        so edge widths are uniform.

        Parameters
        ----------
        main_paper : The paper whose references should be visualised.

        Returns
        -------
        HTML string for embedding via streamlit.components.v1.html().
        """
        net = GraphVisualizer._base_network()

        # ---- Central / focal node ----
        focal_tooltip = (
            f"<b>{main_paper.title}</b><br>"
            f"Authors: {', '.join(main_paper.authors)}<br>"
            f"Field: {main_paper.field_of_study}"
        )
        net.add_node(
            main_paper.title,
            label=GraphVisualizer._truncate(main_paper.title, 40),
            color=GraphVisualizer.COLOR_MAIN_PAPER,
            size=35,
            title=focal_tooltip,
            shape="box",
            borderWidth=3,
        )

        # ---- Reference satellite nodes ----
        for ref in main_paper.references:
            ref_title = ref.get("title", "Unknown")
            ref_year = ref.get("year", "N/A")
            node_id = f"{ref_title} ({ref_year})"

            net.add_node(
                node_id,
                label=GraphVisualizer._truncate(ref_title, 25),
                color=GraphVisualizer.COLOR_REF_NODE,
                size=15,
                title=f"<b>{ref_title}</b><br>Year: {ref_year}",
                shape="ellipse",
            )
            # Uniform edge – no cross-paper strength data available
            net.add_edge(
                main_paper.title,
                node_id,
                color="#999999",
                width=1.5,
                arrows="to",
            )

        GraphVisualizer._apply_physics(net)
        return net.generate_html()


# ===========================================================================
# UTILITY / CORE LOGIC
# ===========================================================================

def parse_pdf(file) -> str:
    """
    Extracts and concatenates text from all pages of an uploaded PDF.

    Parameters
    ----------
    file : File-like object from st.file_uploader.

    Returns
    -------
    Stripped full-text string, or empty string on failure.
    """
    try:
        reader = PdfReader(file)
        return " ".join(
            page.extract_text()
            for page in reader.pages
            if page.extract_text()
        ).strip()
    except Exception as exc:
        st.error(f"Failed to read PDF: {exc}")
        return ""


def save_file_to_server(uploaded_file) -> str:
    """
    Persists an uploaded file to the local 'storage/' directory.

    A UUID prefix prevents filename collisions across users/sessions.

    Parameters
    ----------
    uploaded_file : Streamlit UploadedFile object.

    Returns
    -------
    Relative path string (e.g. 'storage/<uuid>_paper.pdf').
    """
    os.makedirs("storage", exist_ok=True)
    file_path = f"storage/{uuid.uuid4()}_{uploaded_file.name}"
    with open(file_path, "wb") as fh:
        fh.write(uploaded_file.getbuffer())
    return file_path


# ===========================================================================
# RESOURCE CACHING
# ===========================================================================

@st.cache_resource
def get_ai_client() -> genai.Client:
    """
    Initialises the Google GenAI client exactly once per Streamlit session.

    Halts the app if the API key is not configured.
    """
    if not GEMINI_API_KEY:
        st.error("Missing GEMINI_API_KEY in environment.")
        st.stop()
    return genai.Client(api_key=GEMINI_API_KEY)


# ===========================================================================
# USER INTERFACE
# ===========================================================================

def render_legend() -> None:
    """
    Displays a sidebar legend that explains connection-strength visual encodings.

    Should be called once per render_ui() invocation to keep users informed
    about what they are looking at in the graph.
    """
    with st.sidebar:
        st.divider()
        st.subheader("📊 Graph Legend")
        st.markdown(
            """
            **Nodes**
            - 🟦 **Blue box** – paper in your collection
            - 🟠 **Amber ellipse** – external reference only
            - *Node size* – proportional to inbound citations from sibling papers

            **Edges**
            - *Width* – thicker = stronger connection (1–10 px)
            - *Color* – ⬜ light grey (weak) → 🟠 orange (strong)
            - *Label* – `strength: N` on sibling-to-sibling edges
            - Hover an edge for mutual vs. one-way citation info
            """
        )
        st.info("💡 Upload PDFs to populate the active collection.")


def render_results(papers: List[Dict[str, Any]], collection_id: str) -> None:
    """
    Renders the paper metadata panel and the connection-strength graph.

    Parameters
    ----------
    papers        : All documents for the active collection.
    collection_id : Display name of the active collection.
    """
    col_meta, col_graph = st.columns([1, 2])

    # ---- Left column: paper metadata cards ----
    with col_meta:
        st.subheader("📄 Collection Papers")
        for paper in papers:
            with st.expander(GraphVisualizer._truncate(paper["title"], 35)):
                st.write(f"**Field:** {paper.get('field_of_study', 'Unknown')}")
                st.write(
                    f"**Methodologies:** "
                    f"{', '.join(paper.get('methodologies', [])) or 'N/A'}"
                )
                st.write(
                    f"**Type:** {paper.get('paper_type', 'Unknown')} "
                    f"({paper.get('publication_type', 'Unknown')})"
                )
                st.write(f"**Perspective:** {paper.get('perspective', 'Unknown')}")

    # ---- Right column: interactive connection-strength graph ----
    with col_graph:
        st.subheader("🕸️ Connection-Strength Citation Network")

        # Inline metric: number of strong (mutual) connections
        edge_weights = ConnectionStrength.compute_edge_weights(papers)
        mutual_links = sum(1 for w in edge_weights.values() if w == 2)
        one_way_links = sum(1 for w in edge_weights.values() if w == 1)

        m1, m2, m3 = st.columns(3)
        m1.metric("Papers", len(papers))
        m2.metric("Mutual citations", mutual_links, help="Both papers cite each other")
        m3.metric("One-way citations", one_way_links, help="Only one paper cites the other")

        graph_html = GraphVisualizer.generate_comprehensive_graph(papers)
        components.html(graph_html, height=740)


def render_ui() -> None:
    """
    Top-level UI entry point.  Orchestrates sidebar, collection management,
    file upload, AI processing, and results rendering.

    Auth note: user_id is mocked here. In production, replace with a real
    session-based auth lookup (e.g., st.session_state["user_id"]).
    """
    st.title("Research Grapher 🚀")
    st.markdown(
        f"Structure your research citations instantly using **{MODEL_NAME}**. "
        "Connection strength shows how tightly your papers are interlinked."
    )

    # Mock user – swap with real session auth in production
    user_id: str = "user_123"

    # Instantiate service objects once per render cycle
    db_manager = PaperDataManager(MONGO_URI)
    ai_analyzer = AIAnalyzer(GEMINI_API_KEY)

    # ---- Sidebar: workspace / collection selection ----
    with st.sidebar:
        st.header("Workspace")
        existing_collections = db_manager.get_collections(user_id)
        new_collection_name = st.text_input("Create new collection")
        collection_option = st.selectbox(
            "Select collection",
            options=["-- Select --"] + existing_collections,
        )

        # Resolve which collection is active
        if new_collection_name:
            active_collection: Optional[str] = new_collection_name
        elif collection_option != "-- Select --":
            active_collection = collection_option
        else:
            active_collection = None

    # Always render the legend so users know graph encodings before uploading
    render_legend()

    # ---- Main content area ----
    if not active_collection:
        st.info("Please select or create a collection in the sidebar to begin.")
        return

    st.subheader(f"Current Workspace: **{active_collection}**")
    saved_papers = db_manager.get_papers_by_collection(user_id, active_collection)

    st.divider()
    st.subheader("Add Papers to Collection")
    uploaded_files = st.file_uploader(
        "Upload Research PDFs",
        type="pdf",
        accept_multiple_files=True,
    )

    # Process newly uploaded files through AI pipeline
    if uploaded_files:
        with st.spinner("🧠 AI is reading and analysing new papers…"):
            for uploaded_file in uploaded_files:
                file_path = save_file_to_server(uploaded_file)
                raw_text = parse_pdf(uploaded_file)
                extracted_data = ai_analyzer.extract_metadata(raw_text)

                # Construct typed dataclass from AI output
                current_paper = PaperMetadata(
                    title=extracted_data.get("title", "Unknown"),
                    doi=extracted_data.get("doi"),
                    authors=extracted_data.get("authors", []),
                    references=extracted_data.get("references", []),
                    collection_id=active_collection,
                    user_id=user_id,
                    field_of_study=extracted_data.get("field_of_study", "Unknown"),
                    perspective=extracted_data.get("perspective", "Unknown"),
                    methodologies=extracted_data.get("methodologies", []),
                    paper_type=extracted_data.get("paper_type", "Unknown"),
                    publication_type=extracted_data.get("publication_type", "Unknown"),
                    file_path=file_path,
                )
                db_manager.save_paper(current_paper)

        # Re-fetch to include newly saved papers
        saved_papers = db_manager.get_papers_by_collection(user_id, active_collection)

    # ---- Render results or placeholder ----
    if saved_papers:
        st.write(f"Found **{len(saved_papers)}** paper(s) in collection.")
        render_results(saved_papers, active_collection)
    else:
        st.info("No papers saved for this collection yet. Upload a PDF above.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    render_ui()
