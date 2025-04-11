# SHL Assessment Recommendation Tool

A smart AI-powered tool that recommends SHL assessments based on user-provided job descriptions or hiring needs. It uses Sentence Transformers and FAISS for semantic retrieval and Gemini (Google Generative AI) for dynamic summary generation — served via FastAPI backend and Streamlit frontend.

---

## Features

-  Natural language job description input
-  Semantic matching of SHL assessments
-  Fast response using FAISS and MiniLM embeddings
-  Remote and Adaptive support metadata
-  Gemini-powered auto-generated descriptions
-  Clean Streamlit-based web UI

---

##  Tech Stack

| Layer     | Technology                        |
|-----------|-----------------------------------|
| Backend   | FastAPI, Python, FAISS            |
| Frontend  | Streamlit                         |
| Embedding | SentenceTransformers (MiniLM)     |
| LLM API   | Gemini (Google Generative AI)     |
| Tools     | dotenv, pydantic, logging         |

---

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/shl-assessment-recommender.git
cd shl-assessment-recommender
```

### 2. Setup Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Create a `.env` file:

```
GEMINI_API_KEY=your_gemini_api_key
```

---

##  Prepare Catalog & Embeddings

Make sure your `shl_catalog.json` is in the `data/` folder. Then run:

```bash
python -c "from retriever import create_and_save_embeddings; create_and_save_embeddings()"
```

This will generate `cached_embeddings.pkl` file.

---

##  Run Backend (FastAPI)

```bash
uvicorn main:app --reload
```

Check health: [http://localhost:8000/health](http://localhost:8000/health)

---

##  Run Frontend (Streamlit)

```bash
streamlit run app.py
```

Access in browser at: [http://localhost:8501](http://localhost:8501)

---

##  API Endpoints

| Method | Route           | Description                     |
|--------|------------------|---------------------------------|
| GET    | `/health`        | Health check                    |
| POST   | `/recommend`     | Get top SHL assessment matches |


---

##  Example Queries

- "Hiring a Java developer who can collaborate with business teams in 40 minutes."
- "Looking for Python, SQL, and JavaScript test under 60 minutes."
- "Need an analyst assessment with cognitive and personality traits."

---

##  Folder Structure

```
├── app.py                 # Streamlit frontend
├── main.py                # FastAPI backend
├── retriever.py           # Embedding + search logic
├── utils.py               # Helpers & Gemini logic
├── data/
│   ├── shl_catalog.json
│   └── cached_embeddings.pkl
├── .env
├── requirements.txt
└── README.md
```



