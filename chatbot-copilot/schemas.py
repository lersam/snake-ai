"""Shared schemas and constants for the chatbot RAG app.

Contains the fixed category list, configuration values and pydantic models
used by the FastAPI endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel

# Fixed category list requested by the user
CATEGORY_LIST: List[str] = [
    "GameProgrammingBooks",
    "NetworkProgrammingBooks",
    "GuiBooks",
    "JythonBooks",
    "ScientificProgrammingBooks",
    "SystemAdministrationBooks",
    "WebProgrammingBooks",
    "WindowsBooks",
    "XmlBooks",
    "ZopeBooks",
]


@dataclass
class Config:
    data_dir: str = "../chatbot-copilot/data"
    uploads_dir: str = "../chatbot-copilot/uploads"
    chunk_size_words: int = 400
    chunk_overlap_words: int = 50
    top_k: int = 5
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "ollama/llama2"  # override via env or edit if needed


class UploadResponse(BaseModel):
    success: bool
    message: str


class QueryRequest(BaseModel):
    question: str
    max_context: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []

