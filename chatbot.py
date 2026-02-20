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



# Tier 1: Exact phrase matching (expanded with many variations)
_RANK_PHRASES = {
    # ── Popularity ──
    "most popular":    {"col_hints": ["ratings_count", "num_ratings", "popularity", "votes", "reviews"], "order": "desc"},
    "least popular":   {"col_hints": ["ratings_count", "num_ratings", "popularity", "votes", "reviews"], "order": "asc"},
    "most reviewed":   {"col_hints": ["ratings_count", "num_ratings", "reviews", "review_count"],        "order": "desc"},
    "most ratings":    {"col_hints": ["ratings_count", "num_ratings"],                                    "order": "desc"},
    "most reviews":    {"col_hints": ["ratings_count", "num_ratings", "reviews"],                         "order": "desc"},
    # ── Rating ──
    "highest rated":   {"col_hints": ["average_rating", "rating", "score", "stars"], "order": "desc"},
    "highest rating":  {"col_hints": ["average_rating", "rating", "score", "stars"], "order": "desc"},
    "highest ratings": {"col_hints": ["average_rating", "rating", "score", "stars"], "order": "desc"},
    "best rated":      {"col_hints": ["average_rating", "rating", "score", "stars"], "order": "desc"},
    "best rating":     {"col_hints": ["average_rating", "rating", "score", "stars"], "order": "desc"},
    "best ratings":    {"col_hints": ["average_rating", "rating", "score", "stars"], "order": "desc"},
    "top rated":       {"col_hints": ["average_rating", "rating", "score", "stars"], "order": "desc"},
    "top rating":      {"col_hints": ["average_rating", "rating", "score", "stars"], "order": "desc"},
    "lowest rated":    {"col_hints": ["average_rating", "rating", "score", "stars"], "order": "asc"},
    "lowest rating":   {"col_hints": ["average_rating", "rating", "score", "stars"], "order": "asc"},
    "worst rated":     {"col_hints": ["average_rating", "rating", "score", "stars"], "order": "asc"},
    "worst rating":    {"col_hints": ["average_rating", "rating", "score", "stars"], "order": "asc"},
    # ── Pages ──
    "most pages":      {"col_hints": ["num_pages", "pages", "page_count", "length"], "order": "desc"},
    "more pages":      {"col_hints": ["num_pages", "pages", "page_count", "length"], "order": "desc"},
    "highest pages":   {"col_hints": ["num_pages", "pages", "page_count", "length"], "order": "desc"},
    "max pages":       {"col_hints": ["num_pages", "pages", "page_count", "length"], "order": "desc"},
    "fewest pages":    {"col_hints": ["num_pages", "pages", "page_count", "length"], "order": "asc"},
    "fewer pages":     {"col_hints": ["num_pages", "pages", "page_count", "length"], "order": "asc"},
    "least pages":     {"col_hints": ["num_pages", "pages", "page_count", "length"], "order": "asc"},
    "longest book":    {"col_hints": ["num_pages", "pages", "page_count", "length"], "order": "desc"},
    "longest":         {"col_hints": ["num_pages", "pages", "page_count", "length"], "order": "desc"},
    "shortest book":   {"col_hints": ["num_pages", "pages", "page_count", "length"], "order": "asc"},
    "shortest":        {"col_hints": ["num_pages", "pages", "page_count", "length"], "order": "asc"},
    "thickest":        {"col_hints": ["num_pages", "pages", "page_count", "length"], "order": "desc"},
    "thinnest":        {"col_hints": ["num_pages", "pages", "page_count", "length"], "order": "asc"},
    # ── Year ──
    "newest":          {"col_hints": ["published_year", "year", "date", "release"], "order": "desc"},
    "oldest":          {"col_hints": ["published_year", "year", "date", "release"], "order": "asc"},
    "most recent":     {"col_hints": ["published_year", "year", "date", "release"], "order": "desc"},
    "latest":          {"col_hints": ["published_year", "year", "date", "release"], "order": "desc"},
    "earliest":        {"col_hints": ["published_year", "year", "date", "release"], "order": "asc"},
    # ── Price ──
    "most expensive":  {"col_hints": ["price", "cost", "amount"], "order": "desc"},
    "cheapest":        {"col_hints": ["price", "cost", "amount"], "order": "asc"},
    "least expensive": {"col_hints": ["price", "cost", "amount"], "order": "asc"},
}

# Tier 2: Superlative words (including base / comparative forms)
_SUPERLATIVE_HIGH = {"high", "higher", "highest", "best", "top", "most",
                     "maximum", "max", "greatest", "largest", "biggest",
                     "more", "better", "great", "large", "big"}
_SUPERLATIVE_LOW  = {"low", "lower", "lowest", "worst", "least", "minimum",
                     "min", "fewest", "smallest", "fewer", "less", "small",
                     "little"}

# Ordinal words — if a user says "second highest" or "third largest", it's ranking
_ORDINAL_INDICATORS = {"first", "second", "third", "fourth", "fifth",
                       "sixth", "seventh", "eighth", "ninth", "tenth",
                       "1st", "2nd", "3rd", "4th", "5th",
                       "last", "next"}

# Column families: what column-related words map to which column hints
_COLUMN_FAMILIES = {
    "rating":     ["average_rating", "rating", "score", "stars"],
    "ratings":    ["average_rating", "rating", "score", "stars"],
    "rated":      ["average_rating", "rating", "score", "stars"],
    "score":      ["average_rating", "rating", "score", "stars"],
    "stars":      ["average_rating", "rating", "score", "stars"],
    "popular":    ["ratings_count", "num_ratings", "popularity", "votes"],
    "popularity": ["ratings_count", "num_ratings", "popularity", "votes"],
    "reviews":    ["ratings_count", "num_ratings", "reviews", "review_count"],
    "reviewed":   ["ratings_count", "num_ratings", "reviews", "review_count"],
    "page":       ["num_pages", "pages", "page_count", "length"],
    "pages":      ["num_pages", "pages", "page_count", "length"],
    "year":       ["published_year", "year", "date", "release"],
    "recent":     ["published_year", "year", "date", "release"],
    "old":        ["published_year", "year", "date", "release"],
    "new":        ["published_year", "year", "date", "release"],
    "price":      ["price", "cost", "amount"],
    "expensive":  ["price", "cost", "amount"],
    "cheap":      ["price", "cost", "amount"],
    "cost":       ["price", "cost", "amount"],
}


def _find_matching_column(col_hints: list[str], dataset_columns: list[str]) -> str | None:
    """Find the first dataset column that matches any of the hint names."""
    ds_cols_lower = {c.lower(): c for c in dataset_columns}
    for hint in col_hints:
        if hint in ds_cols_lower:
            return ds_cols_lower[hint]
    # Partial match fallback
    for hint in col_hints:
        for col_lower, col_orig in ds_cols_lower.items():
            if hint in col_lower or col_lower in hint:
                return col_orig
    return None


def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (ca != cb)))
        prev = curr
    return prev[len(b)]


def _fuzzy_match_word(word: str, candidates: set[str], max_dist: int = 2) -> str | None:
    """Check if a word fuzzy-matches any candidate (for typos like 'higgest' → 'highest')."""
    if word in candidates:
        return word
    # Only try fuzzy match for words with 4+ chars to avoid false positives
    if len(word) < 4:
        return None
    for candidate in candidates:
        if abs(len(word) - len(candidate)) > max_dist:
            continue
        if _edit_distance(word, candidate) <= max_dist:
            return candidate
    return None


def _is_ranking_query(query: str) -> bool:
    """
    Quick rule-based check: does the query contain ranking/superlative language?
    This is a fast gate so we only call the LLM when needed.
    """
    query_lower = query.lower()
    query_words = query_lower.split()
    words = set(query_words)

    # Check superlative words (direct match)
    if words & _SUPERLATIVE_HIGH or words & _SUPERLATIVE_LOW:
        return True

    # Ordinal + any sort-related word (e.g. "third high", "second largest")
    if words & _ORDINAL_INDICATORS:
        return True

    # Check known ranking phrases
    for phrase in _RANK_PHRASES:
        if phrase in query_lower:
            return True

    # Fuzzy match for typos (e.g. "higgest" → "highest")
    for w in query_words:
        if _fuzzy_match_word(w, _SUPERLATIVE_HIGH) or _fuzzy_match_word(w, _SUPERLATIVE_LOW):
            return True

    return False


def _detect_sort_params_llm(query: str, columns: list[str]) -> tuple[str, str] | None:
    """
    Use the LLM to determine which column to sort by and in which direction.
    Returns (column_name, order) or None.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a data analysis helper. Respond ONLY with valid JSON, nothing else.",
                },
                {
                    "role": "user",
                    "content": (
                        f"A user asked this question about a dataset:\n"
                        f"\"{query}\"\n\n"
                        f"The dataset has these columns:\n{columns}\n\n"
                        f"Which SINGLE numeric column should the data be sorted by to answer "
                        f"this question?\n\n"
                        f"COLUMN SELECTION RULES:\n"
                        f"- Pick the column whose name BEST matches the user's words\n"
                        f"- 'rating count' or 'ratings count' → pick a column with 'count' in name (e.g. ratings_count), NOT average_rating\n"
                        f"- 'rating' or 'rated' (without 'count') → pick average_rating or similar\n"
                        f"- Match the INTENT: if user says 'count', 'number of', 'total' → pick a count/quantity column\n\n"
                        f"SORT DIRECTION RULES:\n"
                        f"- high/highest/top/best/most/largest → \"desc\"\n"
                        f"- low/lowest/worst/least/smallest → \"asc\"\n\n"
                        f"Respond ONLY with JSON: {{\"column\": \"<exact_column_name>\", \"order\": \"desc\" or \"asc\"}}\n"
                        f"The column value MUST exactly match one of the column names listed above."
                    ),
                },
            ],
            temperature=0,
            max_tokens=80,
        )

        text = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        result = json.loads(text)
        col = result.get("column")
        order = result.get("order", "desc")

        if col in columns:
            print(f"🧠 LLM sort detection: column='{col}', order='{order}'")
            return col, order

        # Case-insensitive fallback
        col_map = {c.lower(): c for c in columns}
        if col and col.lower() in col_map:
            actual = col_map[col.lower()]
            print(f"🧠 LLM sort detection (case-fixed): column='{actual}', order='{order}'")
            return actual, order

        print(f"⚠️ LLM returned unknown column: '{col}'")
        return None
    except Exception as e:
        print(f"⚠️ LLM sort detection failed: {e}")
        return None


def _detect_order(query: str) -> str:
    """
    Rule-based sort direction detection — more reliable than LLM for this.
    Returns 'desc' or 'asc'.
    """
    query_lower = query.lower()
    words = set(query_lower.split())

    # Check for explicit low/descending words
    if words & _SUPERLATIVE_LOW:
        return "asc"

    # Check for phrases that clearly mean ascending
    for phrase in _RANK_PHRASES:
        if phrase in query_lower and _RANK_PHRASES[phrase]["order"] == "asc":
            return "asc"

    # Default to descending (most common: "highest", "best", "top", "most", etc.)
    return "desc"


def _get_sorted_records(query: str, n: int = 10) -> tuple[list[dict], str | None, str | None]:
    """
    If the query is a ranking/superlative question, use the LLM to identify
    the correct sort column, then return the actual top/bottom N records.
    Returns: (sorted_records, sort_column, sort_order) or ([], None, None).
    """
    if not _state["ready"] or not _state["books_metadata"]:
        return [], None, None

    # Fast gate: skip LLM call if no superlative/ranking language detected
    if not _is_ranking_query(query):
        return [], None, None

    # LLM determines the exact column (it's great at this)
    result = _detect_sort_params_llm(query, _state["dataset_columns"])
    if not result:
        return [], None, None

    col, llm_order = result

    # Use rule-based direction — more reliable than LLM for sort order
    order = _detect_order(query)
    if order != llm_order:
        print(f"🔄 Overriding LLM order '{llm_order}' → '{order}' (rule-based)")

    ascending = order == "asc"

    df = pd.DataFrame(_state["books_metadata"])
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df_sorted = df.dropna(subset=[col]).sort_values(by=col, ascending=ascending).head(n)

    records = df_sorted.to_dict(orient="records")
    print(f"📊 Ranking query → sorted by '{col}' ({order}), returning {len(records)} records")
    return records, col, order


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
    """Clean a record dict for JSON output — works with any dataset."""
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

    try:
        # Dataset metadata — the LLM needs to know the FULL dataset stats
        ds_name    = _state["dataset_name"] or "Unknown"
        ds_rows    = _state["dataset_rows"]
        ds_columns = _state["dataset_columns"]

        # 1. RETRIEVE — text-similarity search
        retrieved = _retrieve(user_query, top_k=7)

        # 2. RANKING BOOST — for superlative queries, get actual sorted records
        sorted_records, sort_col, sort_order = _get_sorted_records(user_query, n=10)

        # Merge: sorted records first (they are the ground truth for rankings),
        # then any FAISS results not already included
        if sorted_records:
            # Use a set of unique identifiers to avoid duplicates
            seen = set()
            merged = []
            for rec in sorted_records:
                key = tuple(str(rec.get(c, "")) for c in ds_columns[:3])  # identify by first 3 cols
                if key not in seen:
                    seen.add(key)
                    merged.append(rec)
            for rec in retrieved:
                key = tuple(str(rec.get(c, "")) for c in ds_columns[:3])
                if key not in seen:
                    seen.add(key)
                    merged.append(rec)
            final_records = merged
            context_label = f"Records sorted by '{sort_col}' ({sort_order}ending)"
        else:
            final_records = retrieved
            context_label = f"Top {len(retrieved)} most relevant records"

        if not final_records:
            return {
                "response": (
                    "I couldn't find any relevant results for your question. "
                    "Could you rephrase or try different keywords?"
                ),
                "metadata": [],
            }

        # 3. AUGMENT — build context from records
        columns = ds_columns
        context_blocks = []
        for i, book in enumerate(final_records):
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

        # 4. GENERATE — system prompt includes FULL dataset metadata
        sort_note = ""
        if sorted_records and sort_col:
            sort_note = (
                f"\nIMPORTANT: The context below contains records ACTUALLY sorted by '{sort_col}' "
                f"from the FULL dataset of {ds_rows} records. These ARE the real top/bottom records, "
                f"not just text-similar ones. Use them confidently for ranking answers.\n"
            )

        system_prompt = (
            f"You are an expert data assistant for a dataset called '{ds_name}'.\n"
            f"IMPORTANT DATASET FACTS (use these for aggregate/count questions):\n"
            f"  - Dataset name: {ds_name}\n"
            f"  - Total number of records in the FULL dataset: {ds_rows}\n"
            f"  - Columns: {', '.join(ds_columns)}\n"
            f"{sort_note}\n"
            f"RULES:\n"
            f"1. For questions about total counts, totals, or 'how many', use the DATASET FACTS above "
            f"   (total records = {ds_rows}), NOT the number of retrieved context records.\n"
            f"2. Answer using ONLY the dataset facts and context provided. Do NOT invent data.\n"
            f"3. Be warm, conversational, and specific — quote values when relevant.\n"
            f"4. Plain text only, no markdown formatting."
        )

        user_prompt = (
            f"User's Question:\n{user_query}\n\n"
            f"Context — {context_label} (out of {ds_rows} total records):\n"
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
            "metadata": [_clean(b) for b in final_records],
        }

    except Exception as e:
        print(f"❌ Error in answer_question: {e}")
        import traceback
        traceback.print_exc()
        return {
            "response": f"I encountered an error while processing your request: {str(e)}",
            "metadata": [],
        }
