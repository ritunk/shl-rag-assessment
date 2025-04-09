from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import retrieve
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    query: str

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    try:
        query = request.query
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
            
        results = retrieve(query)
        return {"recommended_assessments": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

