# edugen/rag.py  — cloud-friendly, no chroma
from pathlib import Path
from pypdf import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import os, re, uuid, json

# ---- config ----
load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

STORE_DIR = Path("/tmp/edugen_store")  # Streamlit Cloud writable path
STORE_DIR.mkdir(parents=True, exist_ok=True)
STORE_PATH = STORE_DIR / "store.json"
EMB_PATH = STORE_DIR / "embeddings.npy"

def _load_store():
    docs, metas = [], []
    if STORE_PATH.exists():
        data = json.loads(STORE_PATH.read_text())
        docs = data.get("docs", [])
        metas = data.get("metas", [])
    embs = np.load(EMB_PATH) if EMB_PATH.exists() else np.zeros((0, 768), dtype=np.float32)
    return docs, metas, embs

def _save_store(docs, metas, embs):
    STORE_PATH.write_text(json.dumps({"docs": docs, "metas": metas}))
    np.save(EMB_PATH, embs)

def _clean(t): 
    return re.sub(r"\s+", " ", (t or "")).strip()

def pdf_to_chunks(pdf_path: str, chunk=800, overlap=120):
    reader = PdfReader(pdf_path)
    if getattr(reader, "is_encrypted", False):
        try: reader.decrypt("")
        except Exception: pass
    text = ""
    for page in reader.pages:
        try: text += page.extract_text() or " "
        except Exception: continue
    text = _clean(text)
    if not text:
        raise ValueError("No extractable text found (likely a scanned PDF).")
    chunks, i = [], 0
    while i < len(text):
        part = text[i:i+chunk]
        end = part.rfind(". ")
        if end > int(chunk*0.5): part = part[:end+1]
        chunks.append(part)
        i += max(len(part)-overlap, 1)
    return chunks

def _embed_batch(texts):
    vecs = []
    for t in texts:
        e = genai.embed_content(model=EMBED_MODEL, content=t)["embedding"]
        vecs.append(np.array(e, dtype=np.float32))
    return np.vstack(vecs) if vecs else np.zeros((0, 768), dtype=np.float32)

# public API expected by app.py
def upsert_pdf(pdf_path: str, coll_name="edugen"):
    chunks = pdf_to_chunks(pdf_path)
    docs, metas, embs = _load_store()
    new_embs = _embed_batch(chunks)
    embs = np.vstack([embs, new_embs]) if embs.size else new_embs
    docs.extend(chunks)
    metas.extend([{"source": pdf_path}] * len(chunks))
    _save_store(docs, metas, embs)
    return len(chunks)

def retrieve(query: str, k=5, coll_name="edugen"):
    docs, metas, embs = _load_store()
    if len(docs) == 0:
        return [], []
    q = _embed_batch([query])[0]
    denom = (np.linalg.norm(embs, axis=1) * np.linalg.norm(q) + 1e-8)
    scores = (embs @ q) / denom
    idx = np.argsort(-scores)[:k]
    return [docs[i] for i in idx], [metas[i] for i in idx]
