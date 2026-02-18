from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from chatbot import answer_question
import os

app = FastAPI(
    title="Book Chatbot API",
    description="Ask questions about books and get AI-powered answers with metadata.",
    version="1.0.0",
)

# ── Serve static UI files ─────────────────────────────────────────────────────
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    response: str
    metadata: list[dict]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Accept a user question and return:
    - response: human-friendly answer
    - metadata: list of relevant book objects
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = answer_question(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}
