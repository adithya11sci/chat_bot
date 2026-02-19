"""
RAG-based Chatbot with Dynamic Dataset Upload
==============================================
Pipeline:
  1. UPLOAD  – user uploads a CSV via the UI
  2. BUILD   – combine all fields into a rich text doc,
               vectorise with TF-IDF, store in FAISS index
  3. RETRIEVE – embed the user query with the same TF-IDF vectoriser,
                compute cosine similarity, return top-k rows
  4. GENERATE – pass the retrieved context to Groq LLM and get a
                human-friendly answer
"""

import os
import hashlib
import json
import pickle
import numpy as np
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
VECTORIZER_PATH  = os.path.join(DATA_DIR, "tfidf_vectorizer.pkl")
METADATA_PATH    = os.path.join(DATA_DIR, "books_metadata.pkl")
DATASET_INFO_PATH = os.path.join(DATA_DIR, "dataset_info.json")

# ── Groq client ───────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

# ── Mutable runtime state ─────────────────────────────────────────────────────
_state = {
    "faiss_index":      None,
    "tfidf_vectorizer": None,
    "books_metadata":   [],
    "dataset_name":     None,
    "dataset_rows":     0,
    "dataset_columns":  [],
    "dataset_hash":     None,
    "ready":            False,
}


# ── Hashing ───────────────────────────────────────────────────────────────────
def _bytes_hash(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ── Preprocess a DataFrame ────────────────────────────────────────────────────
def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ── Build a combined text doc per row ─────────────────────────────────────────
def _make_doc(row: pd.Series, columns: list) -> str:
    boost_cols = {"title", "subtitle", "authors", "categories", "name",
                  "genre", "director", "movie", "book", "subject"}
    parts = []
    for col in columns:
        val = row.get(col, "")
        if pd.isna(val) or val == "":
            continue
        val_str = str(val).strip()
        if col.lower() in boost_cols:
            parts.append(f"{val_str} {val_str}")
        else:
            parts.append(f"{col}: {val_str}")
    return " | ".join(parts).strip()


# ── Build TF-IDF + FAISS index ────────────────────────────────────────────────
def _build_index(df: pd.DataFrame, dataset_name: str, dataset_hash: str):
    print(f"⏳ Building index for '{dataset_name}' ({len(df)} rows)...")

    columns = df.columns.tolist()
    docs = df.apply(lambda r: _make_doc(r, columns), axis=1).tolist()

    vectorizer = TfidfVectorizer(
        max_features=15_000,
        sublinear_tf=True,
        ngram_range=(1, 1),
        min_df=1,
    )
    tfidf_matrix = vectorizer.fit_transform(docs)
    dense = normalize(tfidf_matrix, norm="l2").toarray().astype(np.float32)

    dim   = dense.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(dense)

    # Persist
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    metadata = df.to_dict(orient="records")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    info = {
        "hash": dataset_hash,
        "name": dataset_name,
        "rows": len(df),
        "columns": columns,
    }
    with open(DATASET_INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)

    # Hot-swap state
    _state["faiss_index"]      = index
    _state["tfidf_vectorizer"] = vectorizer
    _state["books_metadata"]   = metadata
    _state["dataset_name"]     = dataset_name
    _state["dataset_rows"]     = len(df)
    _state["dataset_columns"]  = columns
    _state["dataset_hash"]     = dataset_hash
    _state["ready"]            = True

    print(f"✅ Index built. Vocab: {len(vectorizer.vocabulary_)}, Rows: {len(df)}")


def _load_index_from_disk():
    print("✅ Loading existing index from disk...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    with open(DATASET_INFO_PATH) as f:
        info = json.load(f)

    _state["faiss_index"]      = index
    _state["tfidf_vectorizer"] = vectorizer
    _state["books_metadata"]   = metadata
    _state["dataset_name"]     = info.get("name", "Unknown")
    _state["dataset_rows"]     = info.get("rows", len(metadata))
    _state["dataset_columns"]  = info.get("columns", [])
    _state["dataset_hash"]     = info.get("hash")
    _state["ready"]            = True
    print(f"✅ Loaded '{_state['dataset_name']}' ({_state['dataset_rows']} rows)")


# ── Public: load dataset (called from /upload endpoint) ──────────────────────
def load_dataset(csv_path: str, filename: str, csv_bytes: bytes) -> dict:
    new_hash = _bytes_hash(csv_bytes)

    # Check if same dataset already loaded
    if os.path.exists(DATASET_INFO_PATH):
        with open(DATASET_INFO_PATH) as f:
            info = json.load(f)
        if info.get("hash") == new_hash:
            if not _state["ready"]:
                _load_index_from_disk()
            return {
                "status":  "already_loaded",
                "message": f"'{filename}' is already indexed and ready!",
                "name":    _state["dataset_name"],
                "rows":    _state["dataset_rows"],
                "columns": _state["dataset_columns"],
            }

    # New dataset → preprocess and build
    try:
        df = pd.read_csv(csv_path)
        df = _preprocess(df)
    except Exception as e:
        return {"status": "error", "message": f"Failed to read CSV: {e}"}

    if df.empty:
        return {"status": "error", "message": "The uploaded CSV is empty."}

    _build_index(df, filename, new_hash)

    return {
        "status":  "built",
        "message": f"'{filename}' indexed successfully!",
        "name":    filename,
        "rows":    len(df),
        "columns": df.columns.tolist(),
    }


# ── Public: get current status ────────────────────────────────────────────────
def get_status() -> dict:
    return {
        "ready":   _state["ready"],
        "name":    _state["dataset_name"],
        "rows":    _state["dataset_rows"],
        "columns": _state["dataset_columns"],
    }


# ── Public: get preview rows ─────────────────────────────────────────────────
def get_preview(n: int = 10) -> dict:
    """Return the first n rows of the loaded dataset for UI preview."""
    if not _state["ready"] or not _state["books_metadata"]:
        return {"columns": [], "rows": [], "total": 0, "name": None}

    columns = _state["dataset_columns"]
    rows = []
    for rec in _state["books_metadata"][:n]:
        row = {}
        for col in columns:
            val = rec.get(col, "")
            if isinstance(val, float) and np.isnan(val):
                row[col] = ""
            elif isinstance(val, float) and val == int(val):
                row[col] = int(val)
            else:
                # Truncate long text for preview
                val_str = str(val) if val is not None else ""
                row[col] = val_str[:80] + "…" if len(val_str) > 80 else val_str
        rows.append(row)

    return {
        "columns": columns,
        "rows":    rows,
        "total":   _state["dataset_rows"],
        "name":    _state["dataset_name"],
    }


# ── Startup: load existing index if present ──────────────────────────────────
if all(os.path.exists(p) for p in [FAISS_INDEX_PATH, VECTORIZER_PATH,
                                    METADATA_PATH, DATASET_INFO_PATH]):
    _load_index_from_disk()
else:
    print("📂 No dataset indexed yet. Upload a CSV via the UI to get started.")


# ── RAG Retrieval ─────────────────────────────────────────────────────────────
MIN_SCORE = 0.05

def _retrieve(query: str, top_k: int = 7) -> list[dict]:
    if not _state["ready"]:
        return []

    query_vec   = _state["tfidf_vectorizer"].transform([query])
    query_dense = normalize(query_vec, norm="l2").toarray().astype(np.float32)
    scores, indices = _state["faiss_index"].search(query_dense, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        if float(score) < MIN_SCORE:
            continue
        book = _state["books_metadata"][idx].copy()
        book["_score"] = round(float(score), 4)
        results.append(book)
    return results


# ── Clean dict for JSON output ────────────────────────────────────────────────
def _clean(book: dict) -> dict:
    out = {}
    for k, v in book.items():
        if k.startswith("_"):
            continue
        if isinstance(v, float) and np.isnan(v):
            out[k] = None
        elif isinstance(v, float) and v == int(v):
            out[k] = int(v)
        else:
            out[k] = v
    return out


# ── Main RAG pipeline ────────────────────────────────────────────────────────
def answer_question(user_query: str) -> dict:
    if not _state["ready"]:
        return {
            "response": "No dataset is loaded yet. Please upload a CSV file first.",
            "metadata": [],
        }

    # Dataset metadata — the LLM needs to know the FULL dataset stats
    ds_name    = _state["dataset_name"] or "Unknown"
    ds_rows    = _state["dataset_rows"]
    ds_columns = _state["dataset_columns"]

    # 1. RETRIEVE
    retrieved = _retrieve(user_query, top_k=7)

    if not retrieved:
        return {
            "response": (
                "I couldn't find any relevant results for your question. "
                "Could you rephrase or try different keywords?"
            ),
            "metadata": [],
        }

    # 2. AUGMENT — build context from retrieved rows
    columns = ds_columns
    context_blocks = []
    for i, book in enumerate(retrieved):
        lines = [f"[Record {i+1}]"]
        for col in columns:
            val = book.get(col, "")
            if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
                continue
            val_str = str(val)
            if len(val_str) > 600:
                val_str = val_str[:600] + "…"
            lines.append(f"{col}: {val_str}")
        context_blocks.append("\n".join(lines))

    context = ("\n" + "-" * 60 + "\n").join(context_blocks)

    # 3. GENERATE — system prompt includes FULL dataset metadata
    system_prompt = (
        f"You are an expert data assistant for a dataset called '{ds_name}'.\n"
        f"IMPORTANT DATASET FACTS (use these for aggregate/count questions):\n"
        f"  - Dataset name: {ds_name}\n"
        f"  - Total number of records in the FULL dataset: {ds_rows}\n"
        f"  - Columns: {', '.join(ds_columns)}\n\n"
        f"RULES:\n"
        f"1. For questions about total counts, totals, or 'how many', use the DATASET FACTS above "
        f"   (total records = {ds_rows}), NOT the number of retrieved context records.\n"
        f"2. The context below shows only the TOP {len(retrieved)} most relevant records out of {ds_rows} total.\n"
        f"3. Answer using ONLY the dataset facts and context provided. Do NOT invent data.\n"
        f"4. Be warm, conversational, and specific — quote values when relevant.\n"
        f"5. Plain text only, no markdown formatting."
    )

    user_prompt = (
        f"User's Question:\n{user_query}\n\n"
        f"Context — Top {len(retrieved)} most relevant records (out of {ds_rows} total):\n"
        f"{context}\n\n"
        f"Answer the user's question. Remember: the FULL dataset has {ds_rows} records total."
    )

    llm_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=700,
    ).choices[0].message.content.strip()

    return {
        "response": llm_response,
        "metadata": [_clean(b) for b in retrieved],
    }
