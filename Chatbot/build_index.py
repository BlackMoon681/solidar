"""
chatbot/build_index.py
-----------------------
Reads documents.jsonl, builds FAISS embeddings, saves index + metadata.
Run after data_prep.py.
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_dir      = os.path.dirname(os.path.abspath(__file__))
INPUT     = os.path.join(_dir, "data",  "documents.jsonl")
INDEX_OUT = os.path.join(_dir, "index", "faiss_index.bin")
META_OUT  = os.path.join(_dir, "data",  "documents_with_meta.json")

os.makedirs(os.path.join(_dir, "index"), exist_ok=True)

if not os.path.exists(INPUT):
    raise FileNotFoundError(f"Run data_prep.py first. Missing: {INPUT}")

documents = []
texts     = []

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        documents.append(doc)
        texts.append(doc["text"])

print(f"Loaded {len(texts)} documents")

print("Encoding with all-MiniLM-L6-v2...")
model      = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"FAISS index: {index.ntotal} vectors, dim={embeddings.shape[1]}")

faiss.write_index(index, INDEX_OUT)
with open(META_OUT, "w", encoding="utf-8") as f:
    json.dump(documents, f, ensure_ascii=False)

print(f"Saved: {INDEX_OUT}")
print(f"Saved: {META_OUT}")
