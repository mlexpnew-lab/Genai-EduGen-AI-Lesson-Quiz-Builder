from pathlib import Path
from pypdf import PdfReader
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from dotenv import load_dotenv
import os, uuid, re

load_dotenv(override=True)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")

CHROMA_DIR = Path("/tmp/.chroma")  # Streamlit Cloud writable path
CHROMA_DIR.mkdir(exist_ok=True)

_DB_CLIENT = None

def _client():
    global _DB_CLIENT
    if _DB_CLIENT is None:
        _DB_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(allow_reset=True))
    return _DB_CLIENT

def _clean(t): return re.sub(r"\s+", " ", (t or "")).strip()

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
    if not text: raise ValueError("No extractable text found (likely a scanned PDF).")
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
        vecs.append(genai.embed_content(model=EMBED_MODEL, content=t)["embedding"])
    return vecs

def get_db(name="edugen"):
    c = _client()
    names = [col.name for col in c.list_collections()]
    return c.get_collection(name=name) if name in names else c.create_collection(name=name, metadata={"hnsw:space":"cosine"})

def upsert_pdf(pdf_path: str, coll_name="edugen"):
    chunks = pdf_to_chunks(pdf_path)
    coll = get_db(coll_name)
    ids = [str(uuid.uuid4()) for _ in chunks]
    embeds = _embed_batch(chunks)
    coll.upsert(ids=ids, documents=chunks, embeddings=embeds, metadatas=[{"source": pdf_path}]*len(chunks))
    return len(chunks)

def retrieve(query: str, k=5, coll_name="edugen"):
    coll = get_db(coll_name)
    q = _embed_batch([query])[0]
    r = coll.query(query_embeddings=[q], n_results=k)
    return r["documents"][0], r["metadatas"][0]
