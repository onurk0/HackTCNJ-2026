# HackTCNJ-2026
2026 tcnj hackathon

# How to Run
## Natively
1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`
4. `streamlit run app.py`
## Docker
1. Build the Docker Image
`sudo docker build -t hacktcnj .`
2. Run Docker Container
`docker run -p 8501:8501 --env-file .env hacktcnj`
