"""FastAPI application for the simple RAG chatbot.

Endpoints:
- GET / -> categories page
- GET /chat/{category} -> chat UI
- GET /upload -> upload UI
- POST /upload -> accepts multipart PDF + category and ingests
- POST /chat/{category}/query -> query with retrieval + LLM generation
"""
from __future__ import annotations

import logging
import os

# Try to import FastAPI and helpers; if unavailable in the analysis environment,
# provide lightweight placeholders so static checks don't fail. At runtime the
# real FastAPI package should be installed.
try:
    from fastapi import FastAPI, File, UploadFile, Form, Request
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
except Exception:  # pragma: no cover - fallback for IDE/static analysis
    FastAPI = object
    File = object
    UploadFile = object
    Form = object
    Request = object
    HTMLResponse = object
    JSONResponse = object
    FileResponse = object
    StaticFiles = object

from .schemas import Config, CATEGORY_LIST, QueryRequest, QueryResponse
from .ingest import ingest_pdf_file
from .vector_store import get_store_for_category
from .embedder import get_default_embedder
from .llm import OllamaClient

logger = logging.getLogger("__main__")
logging.basicConfig(level=logging.DEBUG)

cfg = Config()

app = FastAPI(title="Chatbot RAG") if isinstance(FastAPI, type) else None

# ensure directories
os.makedirs(cfg.data_dir, exist_ok=True)
os.makedirs(cfg.uploads_dir, exist_ok=True)

# mount static if FastAPI is available
if app is not None:
    # Safe-guard: if StaticFiles is the placeholder, this will raise at runtime unless FastAPI installed.
    app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    p = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(p, "r", encoding="utf-8") as fh:
        s = fh.read()
    s = s.replace("%CATS_PLACEHOLDER%", str(CATEGORY_LIST))
    return HTMLResponse(content=s)


@app.get("/chat/{category}", response_class=HTMLResponse)
async def chat_page(category: str) -> HTMLResponse:
    if category not in CATEGORY_LIST:
        return HTMLResponse(content=f"Unknown category: {category}", status_code=404)
    p = os.path.join(os.path.dirname(__file__), "static", "chat.html")
    with open(p, "r", encoding="utf-8") as fh:
        s = fh.read()
    return HTMLResponse(content=s)


@app.get("/upload", response_class=HTMLResponse)
async def upload_page() -> HTMLResponse:
    p = os.path.join(os.path.dirname(__file__), "static", "upload.html")
    with open(p, "r", encoding="utf-8") as fh:
        s = fh.read()
    s = s.replace("%CATS_PLACEHOLDER%", str(CATEGORY_LIST))
    return HTMLResponse(content=s)


@app.post("/upload")
async def upload(file: UploadFile = File(...), category: str = Form(...)):
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"success": False, "message": "Only PDF allowed"})
    dst = os.path.join(cfg.uploads_dir, file.filename)
    content = await file.read()
    with open(dst, "wb") as fh:
        fh.write(content)
    try:
        n = ingest_pdf_file(dst, category, cfg)
        return JSONResponse({"success": True, "message": f"Ingested {n} chunks"})
    except Exception as e:
        logger.exception("Ingest failed")
        return JSONResponse({"success": False, "message": str(e)})


@app.post("/chat/{category}/query")
async def query_category(category: str, q: QueryRequest):
    if category not in CATEGORY_LIST:
        return JSONResponse({"error": "unknown category"}, status_code=404)
    store = get_store_for_category(category, cfg.data_dir)
    embedder = get_default_embedder()
    oq = q.question
    q_emb = embedder.get_embedding(oq)
    results = store.search(q_emb, k=cfg.top_k if q.max_context is None else q.max_context)
    context_texts = [r[1]["text"] for r in results]
    prompt = f"Use the following context chunks to answer the question.\n\nContext:\n" + "\n---\n".join(context_texts) + f"\n\nQuestion: {oq}\nAnswer:" if context_texts else f"No context available. Answer concisely.\nQuestion: {oq}\nAnswer:"
    llm = OllamaClient(cfg)
    ans = llm.generate(prompt)
    resp = QueryResponse(answer=ans, sources=[r[1]["text"] for r in results])
    # support pydantic v1 and v2 serialization
    content = resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
    return JSONResponse(content=content)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("chatbot.app:app", host="127.0.0.1", port=8000, reload=True)
