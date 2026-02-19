<![CDATA[# 📚 BookMind — AI-Powered CSV Chatbot

Upload any CSV dataset, and ask questions about it in plain English. BookMind uses **RAG (Retrieval-Augmented Generation)** to find the most relevant rows and generate smart answers using **Groq LLM**.

**Tech:** Python · FastAPI · FAISS · Groq (LLaMA 3.3 70B) · Vanilla JS

---

## ✨ Features

- **Upload Any CSV** — Drop in any `.csv` file, and it gets indexed automatically
- **Ask in Plain English** — No SQL or code needed, just type your question
- **Smart Search** — Uses TF-IDF + FAISS vector search to find relevant records
- **AI Answers** — Groq LLM generates clear, human-friendly responses
- **Result Cards** — See matching records as cards with ratings, year, description, etc.
- **Dataset Preview** — View a table of your uploaded data right in the UI
- **Hot Swap** — Upload a new CSV anytime — the old index is replaced automatically
- **Persistent Index** — Index is saved to disk, so it survives server restarts

---

## 🗂️ Project Structure

```
chat_bot/
├── main.py              # FastAPI server & API routes
├── chatbot.py           # RAG engine (indexing, retrieval, LLM generation)
├── run.py               # Server launcher
├── books_cleaned.csv    # Sample dataset (~2,000 books)
├── requirements.txt     # Python dependencies
├── .env                 # API key (not committed to git)
├── .gitignore
├── test_api.py          # Simple API test script
├── static/
│   ├── index.html       # UI layout
│   ├── style.css        # Styling
│   └── app.js           # Frontend logic (upload, chat, preview)
└── data/                # Auto-generated (gitignored)
    ├── faiss_index.bin       # FAISS vector index
    ├── tfidf_vectorizer.pkl  # TF-IDF model
    ├── books_metadata.pkl    # Row data
    └── dataset_info.json     # Dataset info (name, hash, columns)
```

---

## 🏗️ How It Works — Architecture

```
                         ┌─────────────┐
                         │   Browser   │
                         │  (HTML/JS)  │
                         └──────┬──────┘
                                │  HTTP
                                ▼
                         ┌─────────────┐
                         │   FastAPI   │
                         │  (main.py)  │
                         └──────┬──────┘
                                │
                                ▼
                         ┌─────────────┐
                         │  RAG Engine │
                         │ (chatbot.py)│
                         └──┬───────┬──┘
                            │       │
                   ┌────────┘       └────────┐
                   ▼                         ▼
            ┌─────────────┐          ┌─────────────┐
            │ TF-IDF +    │          │  Groq API   │
            │ FAISS Index │          │ (LLaMA 3.3) │
            └─────────────┘          └─────────────┘
```

### RAG Pipeline (4 Steps)

| Step | What Happens |
|------|-------------|
| **1. Upload** | User uploads a CSV → each row is turned into a text document → TF-IDF vectorizes them → stored in a FAISS index |
| **2. Retrieve** | User asks a question → question is vectorized the same way → FAISS finds the top 7 most similar rows |
| **3. Augment** | The retrieved rows + dataset info (total rows, columns) are packed into a prompt for the LLM |
| **4. Generate** | Groq LLM reads the prompt and writes a natural language answer → returned with the matching records |

**Key Details:**
- Important columns like `title`, `authors`, and `categories` get extra weight in the index
- The LLM knows the **total** dataset size, so it answers "how many" questions correctly
- If the same CSV is uploaded again (same MD5 hash), the existing index is reused

### Workflow — Upload Flow

```
 User selects CSV file
        │
        ▼
 Browser sends file ──▶ POST /upload ──▶ Save to temp file
                                              │
                                              ▼
                                        Read & clean CSV
                                        (strip columns, fill NaN)
                                              │
                                              ▼
                                        Combine each row into
                                        a single text document
                                              │
                                              ▼
                                        TF-IDF vectorize all docs
                                        (15,000 features, L2 norm)
                                              │
                                              ▼
                                        Build FAISS index
                                        (IndexFlatIP)
                                              │
                                              ▼
                                        Save index + metadata
                                        to data/ folder
                                              │
                                              ▼
                                        Return success ──▶ UI enables chat
```

### Workflow — Chat (Q&A) Flow

```
 User types a question
        │
        ▼
 Browser sends JSON ──▶ POST /chat ──▶ Vectorize the question
                                        (same TF-IDF vectorizer)
                                              │
                                              ▼
                                        FAISS search (top 7 matches)
                                        Filter by min score (0.05)
                                              │
                                              ▼
                                        Build prompt:
                                        • System: dataset name, total rows, columns
                                        • User: question + top 7 records as context
                                              │
                                              ▼
                                        Send prompt to Groq API
                                        (LLaMA 3.3 70B, temp=0.3)
                                              │
                                              ▼
                                        Return JSON:
                                        { response: "...", metadata: [...] }
                                              │
                                              ▼
                                        UI shows answer in chat
                                        + result cards in sidebar
```

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/adithya11sci/chat_bot.git
cd chat_bot
pip install -r requirements.txt
```

### 2. Set Your API Key

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free API key at [console.groq.com](https://console.groq.com)

### 3. Run the Server

```bash
python run.py
```

Open **http://localhost:8000** in your browser.

### 4. Upload & Chat

1. Click **Upload Dataset** in the header
2. Select any `.csv` file
3. Wait for indexing to finish
4. Start asking questions!

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the web UI |
| `POST` | `/chat` | Ask a question (JSON body: `{"question": "..."}`) |
| `POST` | `/upload` | Upload a CSV file (multipart form) |
| `GET` | `/dataset/status` | Get loaded dataset info (name, rows, columns) |
| `GET` | `/dataset/preview` | Get first 10 rows of the dataset |
| `GET` | `/health` | Health check |

### Example — Chat

**Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Top 5 highest rated books"}'
```

**Response:**
```json
{
  "response": "Here are the top 5 highest-rated books...",
  "metadata": [
    {
      "title": "Spider's Web",
      "authors": "Agatha Christie",
      "average_rating": 4.9,
      "published_year": 2000,
      "num_pages": 241
    }
  ]
}
```

---

## 📋 Sample Dataset

The included `books_cleaned.csv` has ~2,000 books with these columns:

| Column | Example |
|--------|---------|
| `isbn13` | `9780002261982` |
| `title` | `"Spider's Web"` |
| `subtitle` | `"A Novel"` |
| `authors` | `"Agatha Christie"` |
| `categories` | `"Detective and mystery stories"` |
| `description` | `"A witty mystery..."` |
| `published_year` | `2000` |
| `average_rating` | `4.90` |
| `num_pages` | `241` |
| `ratings_count` | `5164` |

> You can upload **any CSV** — the system auto-detects columns and adapts.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python, FastAPI, Uvicorn |
| **LLM** | Groq Cloud — LLaMA 3.3 70B Versatile |
| **Search** | TF-IDF (scikit-learn) + FAISS (Facebook) |
| **Data** | pandas, NumPy |
| **Frontend** | HTML, CSS, JavaScript (no framework) |
| **Fonts** | Inter, Playfair Display (Google Fonts) |

---

## ⚙️ Configuration

| Setting | Value | Where |
|---------|-------|-------|
| `GROQ_API_KEY` | Your API key | `.env` file |
| LLM Model | `llama-3.3-70b-versatile` | `chatbot.py` |
| TF-IDF vocabulary | 15,000 features | `chatbot.py` |
| Top-K results | 7 | `chatbot.py` |
| Min similarity score | 0.05 | `chatbot.py` |
| LLM temperature | 0.3 | `chatbot.py` |
| Server port | 8000 | `run.py` |

---

## 🧪 Testing

Make sure the server is running, then:

```bash
python test_api.py
```

This sends a few sample questions and prints the answers.

---

## 📜 License

MIT — free to use, modify, and share.

---

**Built with Python, FastAPI, FAISS, and Groq** · _Upload. Ask. Discover._
]]>
