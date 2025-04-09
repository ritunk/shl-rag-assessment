# SHL RAG Assessment

An end-to-end intelligent recommendation system that suggests SHL assessments based on a user's query using Retrieval-Augmented Generation (RAG) techniques.

##  Objective

To build a semantic search system that:

- Accepts a natural language query.
- Retrieves the most relevant SHL assessments using vector similarity.

##  Architecture

###  Data Preprocessing

- Input: SHL catalog (`shl_catalog.json`)
- Text combined from fields like name, description, and metadata using a utility function.

###  Embedding & Indexing

- Model: `all-MiniLM-L6-v2` from `sentence-transformers`
- FAISS (`IndexFlatL2`) used to index the vector embeddings for fast nearest-neighbor search.

###  Backend (FastAPI)

- `/recommend` endpoint that returns top-k most relevant assessments in JSON.
- CORS enabled for cross-origin requests from the frontend.

###  Frontend (Streamlit)

- Input field to type your query.
- Calls the backend and displays recommended assessments.

##  Live Demo

-  **Frontend App:** [Try it Live](https://shl-rag-assessment-etqxhj6p6otetqsvjyca5d.streamlit.app/)
-  **Backend API:** [`/recommend`](https://shl-rag-assessment.onrender.com/recommend?query=problem-solving)
-  **GitHub Code:** [View Repo](https://github.com/ritunk/shl-rag-assessment)

##  Features

-  Fast and scalable FAISS indexing
-  Lazy loading of heavy models
-  Fully deployed backend and frontend
-  Simple and user-friendly UI

##  Setup Instructions

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
streamlit run app.py
```

## Author

Ritik Raj ([@ritunk](https://github.com/ritunk))
Final Year CSE, IIIT Bhagalpur

