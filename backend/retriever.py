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
import pickle
import gc
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from utils import build_search_text, format_response
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals
model = None
index = None
catalog = None

EMBEDDING_PATH = os.path.join(os.path.dirname(__file__), "data/cached_embeddings.pkl")
CATALOG_PATH = os.path.join(os.path.dirname(__file__), "data/shl_catalog.json")

def load_catalog() -> List[Dict[str, Any]]:
    try:
        with open(CATALOG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load catalog: {e}")
        return []

def create_and_save_embeddings():
    global catalog
    logger.info("Creating and caching embeddings...")

    temp_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    if catalog is None:
        catalog = load_catalog()

    corpus = [build_search_text(item) for item in catalog]
    embeddings = temp_model.encode(corpus, show_progress_bar=True, batch_size=4, normalize_embeddings=True)
    
    with open(EMBEDDING_PATH, "wb") as f:
        pickle.dump(embeddings, f)

    del temp_model, corpus, embeddings
    gc.collect()
    logger.info("Embeddings cached.")

def load_index():
    global model, index, catalog

    if index is not None and catalog is not None:
        return

    start = time.time()
    logger.info("Loading index and model...")

    catalog = load_catalog()
    if not os.path.exists(EMBEDDING_PATH):
        create_and_save_embeddings()

    with open(EMBEDDING_PATH, "rb") as f:
        corpus_embeddings = pickle.load(f)

    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(corpus_embeddings))

    del corpus_embeddings
    gc.collect()

    logger.info(f"Index loaded in {time.time() - start:.2f}s")

def retrieve(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        logger.info(f"Running retrieve for query: {query[:50]}...")
        load_index()

        global model
        if model is None:
            model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

        query_embedding = model.encode([query], show_progress_bar=False)
        D, I = index.search(np.array(query_embedding), min(top_k, len(catalog)))

        results = [format_response(catalog[i]) for i in I[0]]
        logger.info(f"Returning {len(results)} formatted results")
        return results[:top_k]

    except Exception as e:
        logger.error(f"Retrieve error: {e}")
        return [format_response(catalog[0])] if catalog else []
