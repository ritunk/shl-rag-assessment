# from sentence_transformers import SentenceTransformer
# import json
# import numpy as np
# import faiss

# # Load model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Load catalog
# with open("data/shl_catalog.json", "r") as f:
#     catalog = json.load(f)

# # Prepare searchable text
# corpus = [
#     f"{item['name']} {item['type']} {item['duration']} Remote: {item['remote']} Adaptive: {item['adaptive']}"
#     for item in catalog
# ]

# # Create embeddings
# corpus_embeddings = model.encode(corpus)

# # Initialize FAISS
# dimension = corpus_embeddings[0].shape[0]
# index = faiss.IndexFlatL2(dimension)
# index.add(np.array(corpus_embeddings))

# # Search function
# def retrieve(query, top_k=10):
#     query_embedding = model.encode([query])
#     D, I = index.search(np.array(query_embedding), top_k)
#     return [catalog[i] for i in I[0]]


from sentence_transformers import SentenceTransformer
import json
import numpy as np
import faiss
from utils import build_search_text

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load catalog
with open("data/shl_catalog.json", "r") as f:
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
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [catalog[i] for i in I[0]]

