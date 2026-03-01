# HackTCNJ-2026
2026 tcnj hackathon

# TODO
- [ ] Ensure that papers that cite each other and cite the same papers are shown as such
- [ ] Make it so that selecting collection in UI actually works
- [ ] Fix Graph to make Circular again
- [ ] Automatically add tags and color paper nodes accordingly
- [ ] Universal Papers Search (by tag, name, author)
- [ ] Paper info sidebar
- [ ] Improve UI Logic

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
