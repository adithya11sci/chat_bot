"""
RAG-based Book Chatbot
======================
Pipeline:
  1. BUILD  (once at startup) – combine all book fields into a rich text doc,
             vectorise with TF-IDF, save the matrix + vectoriser to disk.
  2. RETRIEVE – embed the user query with the same TF-IDF vectoriser,
                compute cosine similarity, return top-k books.
  3. GENERATE – pass the retrieved book context to Groq LLM and get a
                human-friendly answer.
"""

import os
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
BASE_DIR         = os.path.dirname(__file__)
CSV_PATH         = os.path.join(BASE_DIR, "books_cleaned.csv")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.bin")
VECTORIZER_PATH  = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
METADATA_PATH    = os.path.join(BASE_DIR, "books_metadata.pkl")

# ── Groq client ───────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

# ── Load & preprocess CSV ─────────────────────────────────────────────────────
print("📚 Loading CSV dataset...")
df = pd.read_csv(CSV_PATH)
df.columns = [c.strip() for c in df.columns]

TEXT_COLS = ["title", "subtitle", "authors", "categories", "description"]
for col in TEXT_COLS:
    if col in df.columns:
        df[col] = df[col].fillna("")

NUMERIC_COLS = ["average_rating", "num_pages", "ratings_count", "published_year"]
for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"✅ Loaded {len(df)} books.")


# ── Build rich text document per book ────────────────────────────────────────
def _make_doc(row: pd.Series) -> str:
    """
    Concatenate ALL meaningful fields into a single text string per book.
    The richer the text, the better the retrieval quality.
    Title/author/category words are repeated to boost their search weight.
    """
    title      = row.get("title", "")
    subtitle   = row.get("subtitle", "")
    authors    = row.get("authors", "")
    categories = row.get("categories", "")
    desc       = row.get("description", "")

    rating = ""
    year   = ""
    pages  = ""
    if pd.notna(row.get("average_rating")):
        rating = f"rating {row['average_rating']:.1f}"
    if pd.notna(row.get("published_year")):
        year = f"published {int(row['published_year'])}"
    if pd.notna(row.get("num_pages")):
        pages = f"{int(row['num_pages'])} pages"

    # Repeat title + author + categories to give them higher TF-IDF weight
    return (
        f"{title} {title} {subtitle} "
        f"{authors} {authors} "
        f"{categories} {categories} "
        f"{desc} "
        f"{rating} {year} {pages}"
    ).strip()


# ── Build or load FAISS + TF-IDF index ───────────────────────────────────────
def _build_index():
    """Vectorise all book documents and store in a FAISS index."""
    print(f"⏳ Building TF-IDF + FAISS index for {len(df)} books...")

    docs = df.apply(_make_doc, axis=1).tolist()

    vectorizer = TfidfVectorizer(
        max_features=30_000,
        sublinear_tf=True,       # log-normalise term frequencies
        ngram_range=(1, 2),      # unigrams + bigrams for better coverage
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(docs)  # sparse (n_books × vocab)

    # Normalise rows → cosine similarity = inner product
    dense = normalize(tfidf_matrix, norm="l2").toarray().astype(np.float32)

    dim   = dense.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product on L2-normalised vecs = cosine
    index.add(dense)

    # Persist to disk
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    metadata = df.to_dict(orient="records")
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Index built and saved. Vocab size: {len(vectorizer.vocabulary_)}")
    return index, vectorizer, metadata


def _load_index():
    print("✅ Loading existing FAISS index from disk...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, vectorizer, metadata


# Startup: build or load
if (
    os.path.exists(FAISS_INDEX_PATH)
    and os.path.exists(VECTORIZER_PATH)
    and os.path.exists(METADATA_PATH)
):
    faiss_index, tfidf_vectorizer, books_metadata = _load_index()
else:
    faiss_index, tfidf_vectorizer, books_metadata = _build_index()


# ── RAG Retrieval ─────────────────────────────────────────────────────────────
def _retrieve(query: str, top_k: int = 7) -> list[dict]:
    """
    Embed the user query using the same TF-IDF vectoriser and find the
    top-k most semantically similar books via FAISS cosine similarity.
    """
    query_vec = tfidf_vectorizer.transform([query])
    query_dense = normalize(query_vec, norm="l2").toarray().astype(np.float32)

    scores, indices = faiss_index.search(query_dense, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        book = books_metadata[idx].copy()
        book["_score"] = round(float(score), 4)
        results.append(book)
    return results


# ── Clean book dict for JSON output ──────────────────────────────────────────
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


# ── Main RAG pipeline ─────────────────────────────────────────────────────────
def answer_question(user_query: str) -> dict:
    """
    Full RAG pipeline:
      Retrieve → Augment context → Generate answer with Groq LLM.

    Returns:
        {
          "response": <human-friendly answer string>,
          "metadata": [<book dict>, ...]
        }
    """

    # ── 1. RETRIEVE ───────────────────────────────────────────────────────────
    retrieved = _retrieve(user_query, top_k=7)

    if not retrieved:
        return {
            "response": (
                "I'm sorry, I couldn't find any relevant books for your question. "
                "Could you rephrase or try a different topic?"
            ),
            "metadata": [],
        }

    # ── 2. AUGMENT – build context from retrieved books ───────────────────────
    context_blocks = []
    for i, book in enumerate(retrieved):
        desc = str(book.get("description", "")).strip()
        desc_preview = desc[:700] if desc else "No description available."

        block = (
            f"[Book {i+1}]\n"
            f"Title       : {book.get('title', 'N/A')}\n"
            f"Subtitle    : {book.get('subtitle', '') or 'N/A'}\n"
            f"Authors     : {book.get('authors', 'N/A')}\n"
            f"Categories  : {book.get('categories', 'N/A')}\n"
            f"Published   : {int(book['published_year']) if pd.notna(book.get('published_year')) else 'N/A'}\n"
            f"Avg Rating  : {book.get('average_rating', 'N/A')}\n"
            f"Pages       : {int(book['num_pages']) if pd.notna(book.get('num_pages')) else 'N/A'}\n"
            f"Ratings Count: {int(book['ratings_count']) if pd.notna(book.get('ratings_count')) else 'N/A'}\n"
            f"Description : {desc_preview}\n"
        )
        context_blocks.append(block)

    context = "\n" + ("-" * 60) + "\n".join(context_blocks)

    # ── 3. GENERATE – ask Groq LLM to answer from context ────────────────────
    system_prompt = (
        "You are an expert book assistant. "
        "Answer the user's question ONLY using the book information provided in the context below. "
        "Be warm, conversational, and specific — quote descriptions, ratings, authors, or page counts when relevant. "
        "If the user asks about the story/plot/content of a book, use the Description field. "
        "If you cannot find the answer in the context, honestly say so. "
        "Do NOT make up any information. Plain text only, no markdown."
    )

    user_prompt = (
        f"User's Question:\n{user_query}\n\n"
        f"Context – Most Relevant Books from the Dataset:\n"
        f"{context}\n\n"
        "Answer the user's question based solely on the context above."
    )

    llm_response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=700,
    ).choices[0].message.content.strip()

    return {
        "response": llm_response,
        "metadata": [_clean(b) for b in retrieved],
    }
