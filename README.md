# 📚 BookMind — AI-Powered Book Chatbot

A Python-based CSV chatbot that answers natural language questions about a curated book dataset using **Groq LLM** and **FastAPI**.

## ✨ Features

- 🤖 **Natural language understanding** — ask anything about books
- 📊 **Smart filtering** — by rating, author, genre, year, page count
- 🎨 **Beautiful dark UI** — two-panel layout with animated background
- ⚡ **Fast API** — FastAPI backend with JSON responses
- 📦 **Structured output** — every response includes `response` + `metadata`

## 🗂️ Project Structure

```
chat_bot/
├── main.py              # FastAPI app entry point
├── chatbot.py           # Core LLM + CSV logic
├── books_cleaned.csv    # Book dataset (2000+ books)
├── requirements.txt
├── .env                 # API keys (not committed)
└── static/
    ├── index.html       # UI
    ├── style.css        # Dark theme styles
    └── app.js           # Frontend logic
```

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key

Edit `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Run the server

```bash
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## 📡 API Usage

### `POST /chat`

**Request:**
```json
{ "question": "Give me top 5 books with best ratings" }
```

**Response:**
```json
{
  "response": "Here are the top 5 highest-rated books in our dataset...",
  "metadata": [
    {
      "isbn13": 9780002261982,
      "title": "Spider's Web",
      "authors": "Agatha Christie",
      "average_rating": 4.9,
      "num_pages": 241,
      "ratings_count": 5164,
      "published_year": 2000,
      "categories": "Detective and mystery stories",
      "description": "..."
    }
  ]
}
```

### `GET /health`
Returns `{"status": "ok"}`.

## 🧠 How It Works

1. **Intent Extraction** — Groq LLM parses the user's question into structured filters (keyword, author, category, rating range, sort order, top-N).
2. **DataFrame Filtering** — pandas filters and sorts the `books_cleaned.csv` dataset.
3. **Answer Generation** — Groq LLM writes a warm, human-friendly response referencing the retrieved books.
4. **JSON Output** — FastAPI returns `{response, metadata}`.

## 📋 Dataset Columns

| Column | Description |
|--------|-------------|
| `isbn13` | ISBN-13 identifier |
| `title` | Book title |
| `subtitle` | Subtitle |
| `authors` | Author(s) |
| `categories` | Genre/category |
| `description` | Book description |
| `published_year` | Year published |
| `average_rating` | Average Goodreads rating |
| `num_pages` | Number of pages |
| `ratings_count` | Number of ratings |

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI, Uvicorn
- **LLM:** Groq (`llama3-70b-8192`)
- **Data:** pandas
- **Frontend:** Vanilla HTML/CSS/JS (no framework)
