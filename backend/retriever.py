# # import os
# # import json
# # import numpy as np
# # import faiss
# # from sentence_transformers import SentenceTransformer
# # from utils import build_search_text, format_response


# # model = None
# # index = None
# # catalog = None

# # def load_index():
# #     global model, index, catalog

# #     if model is not None and index is not None and catalog is not None:
# #         return

   
# #     model = SentenceTransformer('all-MiniLM-L6-v2')

   
# #     json_path = os.path.join(os.path.dirname(__file__), "data/shl_catalog.json")
# #     with open(json_path, "r") as f:
# #         catalog = json.load(f)

 
# #     corpus = [build_search_text(item) for item in catalog]

  
# #     corpus_embeddings = model.encode(corpus)

   
# #     dimension = corpus_embeddings[0].shape[0]
# #     index = faiss.IndexFlatL2(dimension)
# #     index.add(np.array(corpus_embeddings))

# # def retrieve(query, top_k=10):
# #     """Retrieve top k assessments based on query similarity"""
# #     load_index()
    
 
# #     query_embedding = model.encode([query])
    
   
# #     D, I = index.search(np.array(query_embedding), min(top_k, len(catalog)))
    
   
# #     results = [format_response(catalog[i]) for i in I[0]]
    
# #     return results[:max(1, min(10, len(results)))]  



# import os
# import json
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# from utils import build_search_text, format_response

# model = None
# index = None
# catalog = None

# def load_index():
#     global model, index, catalog

#     if model is not None and index is not None and catalog is not None:
#         return

#     try:
#         print(" Loading SentenceTransformer model...")
#         model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
#         print(" Model loaded")

#         json_path = os.path.join(os.path.dirname(__file__), "data/shl_catalog.json")
#         print(f" Loading catalog from: {json_path}")

#         if not os.path.exists(json_path):
#             raise FileNotFoundError(f" File not found: {json_path}")

#         with open(json_path, "r") as f:
#             catalog = json.load(f)
#         print(f" Loaded {len(catalog)} assessments from catalog")

#         corpus = [build_search_text(item) for item in catalog]
#         print(" Creating embeddings for catalog")
#         corpus_embeddings = model.encode(corpus, show_progress_bar=True, batch_size=2, normalize_embeddings=True)

#         dimension = corpus_embeddings[0].shape[0]
#         index = faiss.IndexFlatL2(dimension)
#         index.add(np.array(corpus_embeddings))
#         print(" FAISS index built successfully")

#     except Exception as e:
#         print(" ERROR in load_index:", str(e))
#         raise

# def retrieve(query, top_k=10):
#     """Retrieve top k assessments based on query similarity"""
#     print(" retrieve() called with query:", query)

#     try:
#         load_index()

#         query_embedding = model.encode([query])
#         D, I = index.search(np.array(query_embedding), min(top_k, len(catalog)))

#         results = [format_response(catalog[i]) for i in I[0]]
#         print(f" Retrieved {len(results)} results")
#         return results[:max(1, min(10, len(results)))]

#     except Exception as e:
#         print(" ERROR in retrieve:", str(e))
#         return []


import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging
import gc  # Garbage collector
import pickle
from typing import List, Dict, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
index = None
catalog = None
embeddings_file = "data/cached_embeddings.pkl"
catalog_file = "data/shl_catalog.json"

def build_search_text(item: Dict[str, Any]) -> str:
    """Create searchable text from catalog item"""
    title = item.get("title", "")
    desc = item.get("description", "")
    comp = " ".join(item.get("competencies", []))
    return f"{title} {desc} {comp}"

def format_response(item: Dict[str, Any]) -> Dict[str, Any]:
    """Format a catalog item for API response"""
    return {
        "id": item.get("id", ""),
        "title": item.get("title", ""),
        "description": item.get("description", ""),
        "competencies": item.get("competencies", []),
        "time": item.get("time", ""),
        "level": item.get("level", ""),
    }

def load_catalog() -> List[Dict[str, Any]]:
    """Load catalog from JSON file"""
    try:
        json_path = os.path.join(os.path.dirname(__file__), catalog_file)
        with open(json_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading catalog: {e}")
        return []

def create_and_save_embeddings():
    """Create and save embeddings to a file to avoid recomputation"""
    global catalog
    
    logger.info("Creating new embeddings file")
    # Load the smallest possible model
    temp_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    
    # Load catalog if not already loaded
    if catalog is None:
        catalog = load_catalog()
    
    # Prepare searchable text
    corpus = [build_search_text(item) for item in catalog]
    
    # Create embeddings in very small batches
    batch_size = 8
    all_embeddings = []
    
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        batch_embeddings = temp_model.encode(batch, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
        # Force garbage collection
        gc.collect()
        
    # Combine embeddings
    corpus_embeddings = np.vstack(all_embeddings)
    
    # Save to file
    embeddings_path = os.path.join(os.path.dirname(__file__), embeddings_file)
    with open(embeddings_path, 'wb') as f:
        pickle.dump(corpus_embeddings, f)
    
    # Clean up to free memory
    del temp_model, corpus, all_embeddings, corpus_embeddings
    gc.collect()
    
    logger.info("Embeddings saved successfully")

def load_index():
    """Load index with minimal memory usage"""
    global model, index, catalog
    
    # If already loaded, return
    if index is not None and catalog is not None:
        return
        
    start_time = time.time()
    logger.info("Loading catalog...")
    
    # Load catalog
    catalog = load_catalog()
    
    # Check if we have pre-computed embeddings
    embeddings_path = os.path.join(os.path.dirname(__file__), embeddings_file)
    
    # If embeddings don't exist, create them (this should be done offline)
    if not os.path.exists(embeddings_path):
        logger.info("No cached embeddings found. Creating new embeddings...")
        create_and_save_embeddings()
    
    # Load cached embeddings
    logger.info("Loading cached embeddings...")
    with open(embeddings_path, 'rb') as f:
        corpus_embeddings = pickle.load(f)
    
    # Initialize simple FAISS index
    logger.info("Creating FAISS index...")
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Uses less memory than IndexIVFFlat
    index.add(np.array(corpus_embeddings))
    
    # Free memory
    del corpus_embeddings
    gc.collect()
    
    logger.info(f"Index loaded in {time.time() - start_time:.2f} seconds")

def simple_keyword_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Simple keyword-based fallback search method"""
    logger.info("Using simple keyword search as fallback")
    
    if catalog is None:
        raise ValueError("Catalog not loaded")
        
    # Simple keyword matching
    query_terms = set(query.lower().split())
    results = []
    
    for item in catalog:
        # Calculate a simple score based on keyword matches
        item_text = build_search_text(item).lower()
        score = sum(1 for term in query_terms if term in item_text)
        
        if score > 0:
            results.append((score, item))
    
    # Sort by score descending
    results.sort(reverse=True)
    
    # Return top k formatted results
    return [format_response(item) for _, item in results[:top_k]]

def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieve top k assessments based on query similarity"""
    try:
        logger.info(f"Starting retrieval for query: {query[:50]}...")
        
        # Try to load index
        try:
            # Load catalog and index
            load_index()
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            # Fall back to simple keyword search
            return simple_keyword_search(query, top_k)
        
        # If all goes well, load the model only when needed
        try:
            # Load the model if needed
            global model
            if model is None:
                logger.info("Loading model...")
                # Use the smallest viable model
                model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
                
            # Get query embedding
            logger.info("Generating query embedding")
            query_embedding = model.encode([query], show_progress_bar=False)
            
            # Search in the index
            logger.info("Searching in the index")
            D, I = index.search(np.array(query_embedding), min(top_k, len(catalog)))
            
            # Format results
            results = [format_response(catalog[i]) for i in I[0]]
            
            logger.info(f"Found {len(results)} results")
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Error during embedding search: {e}")
            # Fall back to simple keyword search
            return simple_keyword_search(query, top_k)
            
    except Exception as e:
        logger.error(f"Critical error in retrieve function: {e}")
        # Return at least something as a last resort
        if catalog and len(catalog) > 0:
            return [format_response(catalog[0])]  # Return first catalog item
        return []  # Empty result if nothing else works