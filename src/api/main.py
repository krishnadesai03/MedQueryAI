"""
FastAPI application — Healthcare RAG Chatbot

Endpoints:
  GET  /           → serves the chat UI (index.html)
  POST /ask        → takes a question, returns a cited answer
  GET  /health     → health check
"""
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.generator import RAGGenerator

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Healthcare RAG API",
    description="Answers clinical questions with cited sources from MIMIC-IV data.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialise retriever and generator once at startup (not per request)
retriever: Optional[HybridRetriever] = None
generator: Optional[RAGGenerator] = None


@app.on_event("startup")
def load_models():
    global retriever, generator
    print("Loading retriever...")
    retriever = HybridRetriever()
    print("Loading generator...")
    generator = RAGGenerator()
    print("Ready.")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    idx: int
    disease: str
    doc_type: str
    file_name: str
    note_id: str
    rerank_score: Optional[float]

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    refused: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": retriever is not None}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = retriever.retrieve_with_context(question)
    response = generator.answer(question, result["documents"])

    return AskResponse(
        question=response.question,
        answer=response.answer,
        sources=[SourceItem(**s) for s in response.sources],
        refused=response.refused,
    )
