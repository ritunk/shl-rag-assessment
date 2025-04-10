# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from retriever import retrieve
# from dotenv import load_dotenv
# import os

# load_dotenv()  # Load .env file

# app = FastAPI()

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class RecommendationRequest(BaseModel):
#     query: str

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}

# @app.post("/recommend")
# async def recommend(request: RecommendationRequest):
#     try:
#         print("/recommend called")
#         print("Query received:", request.query)
#         query = request.query
#         if not query:
#             raise HTTPException(status_code=400, detail="Query cannot be empty")
            
#         results = retrieve(query)
#         print(" Retrieved results:", results)
#         return {"recommended_assessments": results}
#     except Exception as e:
#         print("ERROR in /recommend:", str(e))
#         raise HTTPException(status_code=500, detail=str(e))
    

# @app.get("/test-file")
# async def test_file():
#     import os
#     json_path = os.path.join(os.path.dirname(__file__), "data/shl_catalog.json")
#     return {
#         "path_checked": json_path,
#         "file_exists": os.path.exists(json_path)
#     }


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from retriever import retrieve, load_catalog
from dotenv import load_dotenv
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # Load .env file

app = FastAPI()

# Simplified CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    query: str

@app.get("/health")
async def health_check():
    logger.info("Health check called")
    return {"status": "healthy"}

@app.post("/recommend")
async def recommend(request: RecommendationRequest):
    try:
        logger.info(f"/recommend called with query: {request.query[:50]}...")
        start_time = time.time()
        
        query = request.query
        if not query:
            logger.warning("Empty query received")
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Try to get results with timeout protection
        results = retrieve(query, top_k=5)  # Limit to 5 results to reduce processing
        
        logger.info(f"Retrieved {len(results)} results in {time.time() - start_time:.2f} seconds")
        
        # Always return at least one result
        if not results:
            logger.warning("No results found, returning default response")
            catalog = load_catalog()
            if catalog and len(catalog) > 0:
                from retriever import format_response
                results = [format_response(catalog[0])]  # Return first catalog item as fallback
            else:
                results = [{
                    "id": "default",
                    "title": "General Assessment",
                    "description": "A general assessment option when no specific match is found.",
                    "competencies": ["general"],
                    "time": "30 min",
                    "level": "All levels",
                }]
        
        return {"recommended_assessments": results}
    except Exception as e:
        logger.error(f"ERROR in /recommend: {str(e)}", exc_info=True)
        # Return a helpful error message
        raise HTTPException(
            status_code=500, 
            detail="We encountered an issue processing your request. Please try a simpler query or try again later."
        )

@app.get("/test-file")
async def test_file():
    try:
        import os
        json_path = os.path.join(os.path.dirname(__file__), "data/shl_catalog.json")
        file_exists = os.path.exists(json_path)
        logger.info(f"Checking file at {json_path}, exists: {file_exists}")
        
        # Also check for cached embeddings file
        embeddings_path = os.path.join(os.path.dirname(__file__), "data/cached_embeddings.pkl")
        embeddings_exist = os.path.exists(embeddings_path)
        
        return {
            "path_checked": json_path,
            "file_exists": file_exists,
            "embeddings_path": embeddings_path,
            "embeddings_exist": embeddings_exist
        }
    except Exception as e:
        logger.error(f"Error in test-file endpoint: {str(e)}", exc_info=True)
        return {"error": str(e)}


