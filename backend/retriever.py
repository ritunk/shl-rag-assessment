import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from utils import build_search_text

# Global variables for lazy loading
model = None
index = None
catalog = None

def load_index():
    global model, index, catalog

    # Skip if already loaded
    if model is not None and index is not None and catalog is not None:
        return

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load catalog safely
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, "data/shl_catalog.json")
    with open(json_path, "r") as f:
        catalog = json.load(f)

    # Prepare searchable text
    corpus = [build_search_text(item) for item in catalog]

    # Create embeddings
    corpus_embeddings = model.encode(corpus)

    # Initialize FAISS
    dimension = corpus_embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(corpus_embeddings))


# Search function
def retrieve(query, top_k=10):
    load_index()
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [catalog[i] for i in I[0]]
