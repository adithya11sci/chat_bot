import os
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from chatbot import answer_question, load_dataset, get_status

BASE_DIR = os.path.dirname(__file__)

app = FastAPI(
    title="BookMind Chatbot API",
    description="Upload any CSV dataset and ask AI-powered questions.",
    version="2.0.0",
)

# ── Serve static UI files ─────────────────────────────────────────────────────
STATIC_DIR = os.path.join(BASE_DIR, "static")
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
    """Accept a user question, return AI answer + relevant records."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        result = answer_question(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_dataset_endpoint(file: UploadFile = File(...)):
    """Upload a CSV file → build TF-IDF + FAISS index → enable chat."""
    fname = file.filename or "upload.csv"
    if not fname.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        csv_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if len(csv_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, dir=os.path.join(BASE_DIR, "data")
        ) as tmp:
            tmp.write(csv_bytes)
            tmp_path = tmp.name

        result = load_dataset(
            csv_path=tmp_path,
            filename=fname,
            csv_bytes=csv_bytes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return JSONResponse(content=result, status_code=200)


@app.get("/dataset/status")
async def dataset_status():
    """Return info about the currently loaded dataset."""
    return get_status()


@app.get("/health")
async def health():
    info = get_status()
    return {"status": "ok", "dataset_ready": info["ready"]}
