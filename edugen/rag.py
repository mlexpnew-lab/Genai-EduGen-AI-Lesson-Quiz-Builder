# edugen/rag.py
from pathlib import Path
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid, re

CHROMA_DIR = Path(".chroma")
CHROMA_DIR.mkdir(exist_ok=True)

_EMB = None
_DB_CLIENT = None

def _embedder():
    global _EMB
    if _EMB is None:
        # lazy-load to avoid import-time crash
        _EMB = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB

def _client():
    global _DB_CLIENT
    if _DB_CLIENT is None:
        _DB_CLIENT = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(allow_reset=True)
        )
    return _DB_CLIENT

def _clean(t):
    return re.sub(r"\s+", " ", (t or "")).strip()

def pdf_to_chunks(pdf_path: str, chunk=800, overlap=120):
    reader = PdfReader(pdf_path)
    if getattr(reader, "is_encrypted", False):
        try:
            reader.decrypt("")
        except Exception:
            pass

    text = ""
    for page in reader.pages:
        try:
            text += page.extract_text() or " "
        except Exception:
            continue

    text = _clean(text)
    if not text:
        raise ValueError("No extractable text found (likely a scanned PDF).")

    chunks, i = [], 0
    while i < len(text):
        part = text[i:i+chunk]
        end = part.rfind(". ")
        if end > int(chunk * 0.5):
            part = part[:end+1]
        chunks.append(part)
        i += max(len(part) - overlap, 1)
    return chunks

def get_db(name="edugen"):
    client = _client()
    names = [c.name for c in client.list_collections()]
    if name not in names:
        return client.create_collection(name=name, metadata={"hnsw:space":"cosine"})
    return client.get_collection(name=name)

def upsert_pdf(pdf_path: str, coll_name="edugen"):
    chunks = pdf_to_chunks(pdf_path)
    coll = get_db(coll_name)
    ids = [str(uuid.uuid4()) for _ in chunks]
    embeds = _embedder().encode(chunks, convert_to_numpy=True).tolist()
    coll.upsert(ids=ids, documents=chunks, embeddings=embeds,
                metadatas=[{"source": pdf_path}]*len(chunks))
    return len(chunks)

def retrieve(query: str, k=5, coll_name="edugen"):
    coll = get_db(coll_name)
    q_emb = _embedder().encode([query], convert_to_numpy=True).tolist()
    r = coll.query(query_embeddings=q_emb, n_results=k)
    return r["documents"][0], r["metadatas"][0]
