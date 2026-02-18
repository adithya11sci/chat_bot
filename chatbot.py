import os
import json
import re
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Load dataset once at startup ──────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "books_cleaned.csv")
df = pd.read_csv(CSV_PATH)

# Normalise column names (strip whitespace)
df.columns = [c.strip() for c in df.columns]

# Fill NaN with empty string for text columns, keep numeric as-is
TEXT_COLS = ["title", "subtitle", "authors", "categories", "description"]
for col in TEXT_COLS:
    if col in df.columns:
        df[col] = df[col].fillna("")

NUMERIC_COLS = ["average_rating", "num_pages", "ratings_count", "published_year"]
for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ── Groq client ───────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama3-70b-8192"

# ── Helper: build a compact dataset summary for the LLM ──────────────────────
def _dataset_summary() -> str:
    return (
        f"Dataset has {len(df)} books with columns: "
        f"isbn13, title, subtitle, authors, categories, description, "
        f"published_year, average_rating, num_pages, ratings_count.\n"
        f"Rating range: {df['average_rating'].min():.2f} – {df['average_rating'].max():.2f}. "
        f"Year range: {int(df['published_year'].min())} – {int(df['published_year'].max())}."
    )


# ── Helper: search books by keyword / filters ─────────────────────────────────
def _search_books(
    keyword: str = "",
    author: str = "",
    category: str = "",
    min_rating: float = 0.0,
    max_rating: float = 5.0,
    top_n: int = 10,
    sort_by: str = "average_rating",
    ascending: bool = False,
) -> pd.DataFrame:
    result = df.copy()

    if keyword:
        kw = keyword.lower()
        mask = (
            result["title"].str.lower().str.contains(kw, na=False)
            | result["description"].str.lower().str.contains(kw, na=False)
            | result["categories"].str.lower().str.contains(kw, na=False)
        )
        result = result[mask]

    if author:
        result = result[result["authors"].str.lower().str.contains(author.lower(), na=False)]

    if category:
        result = result[result["categories"].str.lower().str.contains(category.lower(), na=False)]

    result = result[
        (result["average_rating"] >= min_rating) & (result["average_rating"] <= max_rating)
    ]

    if sort_by in result.columns:
        result = result.sort_values(sort_by, ascending=ascending)

    return result.head(top_n)


# ── Helper: convert a DataFrame to a list of clean dicts ─────────────────────
def _df_to_metadata(frame: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in frame.iterrows():
        record = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                record[col] = None
            elif isinstance(val, float) and val == int(val):
                record[col] = int(val)
            else:
                record[col] = val
        records.append(record)
    return records


# ── Core: parse intent & answer ───────────────────────────────────────────────
def answer_question(user_query: str) -> dict:
    """
    Returns {"response": str, "metadata": [book_dict, ...]}
    """
    summary = _dataset_summary()

    # ── Step 1: ask the LLM to extract structured intent ─────────────────────
    intent_prompt = f"""You are a book-data assistant. The user asked:
"{user_query}"

Dataset info: {summary}

Extract the user's intent as a JSON object with these optional keys:
- "keyword": string to search in title/description/categories
- "author": author name to filter by
- "category": genre/category to filter by
- "min_rating": minimum average_rating (float)
- "max_rating": maximum average_rating (float)
- "top_n": how many books to return (int, default 5)
- "sort_by": column to sort by — one of: average_rating, ratings_count, published_year, num_pages
- "ascending": true/false (default false = descending)
- "intent_summary": one short sentence describing what the user wants

Return ONLY valid JSON, no markdown fences, no extra text."""

    intent_raw = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": intent_prompt}],
        temperature=0.1,
        max_tokens=300,
    ).choices[0].message.content.strip()

    # Strip markdown fences if the model adds them anyway
    intent_raw = re.sub(r"^```[a-z]*\n?", "", intent_raw)
    intent_raw = re.sub(r"\n?```$", "", intent_raw)

    try:
        intent = json.loads(intent_raw)
    except json.JSONDecodeError:
        intent = {}

    # ── Step 2: retrieve matching books ──────────────────────────────────────
    top_n = int(intent.get("top_n", 5))
    books_df = _search_books(
        keyword=intent.get("keyword", ""),
        author=intent.get("author", ""),
        category=intent.get("category", ""),
        min_rating=float(intent.get("min_rating", 0.0)),
        max_rating=float(intent.get("max_rating", 5.0)),
        top_n=top_n,
        sort_by=intent.get("sort_by", "average_rating"),
        ascending=bool(intent.get("ascending", False)),
    )

    metadata = _df_to_metadata(books_df)

    # ── Step 3: generate a human-friendly answer ──────────────────────────────
    if metadata:
        books_text = "\n".join(
            f"{i+1}. \"{b['title']}\" by {b['authors']} "
            f"(Rating: {b['average_rating']}, Year: {b['published_year']}, "
            f"Pages: {b['num_pages']}, Ratings count: {b['ratings_count']})"
            for i, b in enumerate(metadata)
        )
    else:
        books_text = "No matching books found."

    answer_prompt = f"""You are a friendly book recommendation assistant.

The user asked: "{user_query}"

Here are the relevant books from the dataset:
{books_text}

Write a warm, helpful, conversational response (2–4 sentences) that directly answers the user's question using the books above. 
If no books were found, say so politely and suggest rephrasing.
Do NOT use markdown. Do NOT list the books again — just reference them naturally in your answer."""

    human_response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0.7,
        max_tokens=400,
    ).choices[0].message.content.strip()

    return {"response": human_response, "metadata": metadata}
