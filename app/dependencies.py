"""
app/dependencies.py  —  Shared resources loaded once at startup.

Imports of this module trigger model + chatbot loading.
All route modules import from here to share the same loaded objects.
"""
import os, json
from pathlib import Path

import joblib
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# ML model
# ---------------------------------------------------------------------------
for _model_name in ("risk_model.pkl", "budget_model_v13.pkl", "budget_model_v12.pkl"):
    _model_path = ROOT / "ml" / "model" / _model_name
    if _model_path.exists():
        break
else:
    raise FileNotFoundError("No model found in ml/model/. Run: python ml/train.py")

_saved     = joblib.load(_model_path)
model      = _saved["model"]
FEATURES   = _saved["features"]
CALIBRATOR = _saved.get("calibrator", None)
OPT_THRESH = _saved.get("opt_threshold", 0.5)
FREQ_MAPS  = _saved.get("freq_maps", {})
MODEL_NAME = _model_name

print(f"Model loaded: {MODEL_NAME}  ({len(FEATURES)} features)")

# ---------------------------------------------------------------------------
# Chatbot resources
# ---------------------------------------------------------------------------
_INDEX_PATH = ROOT / "chatbot" / "index" / "faiss_index.bin"
_META_PATH  = ROOT / "chatbot" / "data"  / "documents_with_meta.json"

print("Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
if not _INDEX_PATH.exists():
    raise FileNotFoundError(f"Missing FAISS index: {_INDEX_PATH}")
faiss_index = faiss.read_index(str(_INDEX_PATH))

print("Loading documents...")
if not _META_PATH.exists():
    raise FileNotFoundError(f"Missing metadata: {_META_PATH}")
with open(_META_PATH, "r", encoding="utf-8") as _f:
    documents = json.load(_f)

print(f"Chatbot ready  ({len(documents)} docs)")

# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------
GROQ_MODEL = "llama-3.3-70b-versatile"


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in .env")
    return Groq(api_key=api_key)
