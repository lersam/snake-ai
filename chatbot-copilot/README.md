# Chatbot RAG (FastAPI)

Minimal Retrieval-Augmented Generation demo using Ollama (local HTTP API).

- Two pages: categories list and upload page. Chat is available per category.
- Upload PDFs and assign to one of the fixed categories. The PDF is converted to text, chunked and embedded; vectors are stored under `chatbot/data/`.
- When you ask a question in a category, the app retrieves top-k chunks and calls Ollama to generate an answer including the retrieved context.

Quick start (Windows PowerShell):

1. Create and activate venv (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Start the app:

```powershell
uvicorn chatbot-copilot.app:app --reload --port 8000
```

3. Open http://127.0.0.1:8000/

Notes:
- Ollama must be running locally for LLM answers. If not available the app will return a deterministic fallback string.
- This is a minimal demo: FAISS persistence and advanced error handling are left as future improvements.
