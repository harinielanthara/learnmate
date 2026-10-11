import httpx
from ..config import get_settings
from .base import LLMProvider


class LocalOllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self):
        s = get_settings()
        self.url, self.model = s.ollama_url, s.ollama_model

    def generate(self, messages, temperature=0.1, max_tokens=512, json_mode=False):
        payload = {
            "model": self.model, "messages": messages, "stream": False,
            # num_ctx 4096 keeps the KV cache small: 3B Q4 ≈ 2 GB weights + ~0.5 GB cache
            "options": {"temperature": temperature, "num_ctx": 4096, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        r = httpx.post(f"{self.url}/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()


class APIProvider(LLMProvider):
    """Any OpenAI-compatible endpoint (Groq, OpenAI, OpenRouter...).
    Add an AnthropicProvider later if you want it; interface is identical."""
    name = "api"

    def __init__(self):
        s = get_settings()
        if not s.api_key:
            raise RuntimeError("API_KEY not set")
        self.base, self.key, self.model = s.api_base_url, s.api_key, s.api_model

    def generate(self, messages, temperature=0.1, max_tokens=512, json_mode=False):
        payload = {"model": self.model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        r = httpx.post(f"{self.base}/chat/completions", json=payload, timeout=120,
                       headers={"Authorization": f"Bearer {self.key}"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


def get_llm() -> LLMProvider:
    return APIProvider() if get_settings().llm_provider == "api" else LocalOllamaProvider()
