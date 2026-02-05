"""Simple Ollama HTTP client with fallback response."""
from __future__ import annotations

import json
import logging
import requests

from .schemas import Config

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, cfg: Config):
        self.base = cfg.ollama_url.rstrip("/")
        self.model = cfg.ollama_model

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        url = f"{self.base}/api/generate"
        payload = {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
        try:
            r = requests.post(url, json=payload, timeout=10)
            r.raise_for_status()
            data = r.json()
            # Ollama responses may contain 'response' or different shapes; try common keys
            if isinstance(data, dict) and "response" in data:
                return str(data["response"])
            if isinstance(data, dict) and "text" in data:
                return str(data["text"])  # fallback
            # else convert to string
            return json.dumps(data)
        except Exception as e:
            logger.warning("Ollama generate failed: %s", e)
            # deterministic fallback: echo prompt summary
            return "[OLLAMA_UNAVAILABLE] " + (prompt[:1000] + "..." if len(prompt) > 1000 else prompt)
