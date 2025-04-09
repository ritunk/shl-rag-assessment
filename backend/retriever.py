import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils import build_search_text, format_response


model = None
index = None
catalog = None

def load_index():
    global model, index, catalog

    if model is not None and index is not None and catalog is not None:
        return

   
    model = SentenceTransformer('all-MiniLM-L6-v2')

   
    json_path = os.path.join(os.path.dirname(__file__), "data/shl_catalog.json")
    with open(json_path, "r") as f:
        catalog = json.load(f)

 
    corpus = [build_search_text(item) for item in catalog]

  
    corpus_embeddings = model.encode(corpus)

   
    dimension = corpus_embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(corpus_embeddings))

def retrieve(query, top_k=10):
    """Retrieve top k assessments based on query similarity"""
    load_index()
    
 
    query_embedding = model.encode([query])
    
   
    D, I = index.search(np.array(query_embedding), min(top_k, len(catalog)))
    
   
    results = [format_response(catalog[i]) for i in I[0]]
    
    return results[:max(1, min(10, len(results)))]  

