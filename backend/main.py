from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from retriever import retrieve

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/recommend")
def recommend(query: str):
    try:
        results = retrieve(query)
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}
