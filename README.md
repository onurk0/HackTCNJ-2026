# HackTCNJ-2026

# Linked Lit!
Linked Lit is an AI-powered workspace that tranforms your PDF libraries into a connected knowledge graph.

We built this with researches in mind who may not see the initial connection between their scientifc and research articles.

**The Problem** 
  Researching a new field often times results in an endless spiral of research papers and articles that make no sense. Traditional tools, like Google Scholar, will tell you who cited whom, but there is no context about the citation, leading to work to figure out the use case of each citation.


**Our Solution**
  Lumen Lit automates the boring part of reading research papers - How are these connected? 

  By extracting metadata, visualizing connections, and saving your work we are able to identify the methodologies, research fields, and generate a relationship graph all while saving your work to a stable database so you can continue to work and contribute without worry.

**Tech Stack**
  LLM: Gemini 2.5 Flash known for its good price-to-performance and being overall well-rounded
  Database: MongoDB
  Frontend: Streamlit for its responsive and data focused UI
  Visualization: Pyvis for browser-based interactive graphs
  Infrastructe: Hosted on a Vultr VPS for availability, scalability, and performance.


# TODO
- [ ] AI-based analysis to show how strongly connected papers are
- [ ] Universal Papers Search (by tag, name, author)
- [x] Ensure that papers that cite each other and cite the same papers are shown as such
- [x] Fix Graph to make Circular again
- [x] Improve UI Logic
- [x] Automatically add tags and color paper nodes accordingly
- [x] Make it so that selecting collection in UI actually works
- [x] Paper info sidebar

# How to Run
## Natively
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `streamlit run app.py --server.address 0.0.0.0 --server.port 8501`
## Docker
1. Build the Docker Image
`sudo docker build -t hacktcnj .`
2. Run Docker Container
`docker run -p 8501:8501 --env-file .env hacktcnj`
