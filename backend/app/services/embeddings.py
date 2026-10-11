from functools import lru_cache
from sentence_transformers import SentenceTransformer
from ..config import get_settings

# bge-v1.5 retrieval convention: instruction prefix on QUERIES only, not passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _model() -> SentenceTransformer:
    # CPU on purpose: ~130 MB model, leaves all 8 GB VRAM for the LLM.
    return SentenceTransformer(get_settings().embed_model, device="cpu")


def embed_passages(texts: list[str]) -> list[list[float]]:
    v = _model().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    return v.tolist()


def embed_query(q: str) -> list[float]:
    return _model().encode(QUERY_PREFIX + q, normalize_embeddings=True).tolist()


@lru_cache
def tokenizer():
    return _model().tokenizer
