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

   
#     model = SentenceTransformer('all-MiniLM-L6-v2')

   
#     json_path = os.path.join(os.path.dirname(__file__), "data/shl_catalog.json")
#     with open(json_path, "r") as f:
#         catalog = json.load(f)

 
#     corpus = [build_search_text(item) for item in catalog]

  
#     corpus_embeddings = model.encode(corpus)

   
#     dimension = corpus_embeddings[0].shape[0]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(np.array(corpus_embeddings))

# def retrieve(query, top_k=10):
#     """Retrieve top k assessments based on query similarity"""
#     load_index()
    
 
#     query_embedding = model.encode([query])
    
   
#     D, I = index.search(np.array(query_embedding), min(top_k, len(catalog)))
    
   
#     results = [format_response(catalog[i]) for i in I[0]]
    
#     return results[:max(1, min(10, len(results)))]  



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

    try:
        print("📦 Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Model loaded")

        json_path = os.path.join(os.path.dirname(__file__), "data/shl_catalog.json")
        print(f"📂 Loading catalog from: {json_path}")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"❌ File not found: {json_path}")

        with open(json_path, "r") as f:
            catalog = json.load(f)
        print(f"✅ Loaded {len(catalog)} assessments from catalog")

        corpus = [build_search_text(item) for item in catalog]
        print("🧠 Creating embeddings for catalog")
        corpus_embeddings = model.encode(corpus, show_progress_bar=True)

        dimension = corpus_embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(corpus_embeddings))
        print("✅ FAISS index built successfully")

    except Exception as e:
        print("❌ ERROR in load_index:", str(e))
        raise

def retrieve(query, top_k=10):
    """Retrieve top k assessments based on query similarity"""
    print("🔎 retrieve() called with query:", query)

    try:
        load_index()

        query_embedding = model.encode([query])
        D, I = index.search(np.array(query_embedding), min(top_k, len(catalog)))

        results = [format_response(catalog[i]) for i in I[0]]
        print(f"✅ Retrieved {len(results)} results")
        return results[:max(1, min(10, len(results)))]

    except Exception as e:
        print("❌ ERROR in retrieve:", str(e))
        return []
