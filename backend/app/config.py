from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://learnmate:learnmate@localhost:5432/learnmate"

    llm_provider: str = "ollama"  # ollama | api
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b-instruct-q4_K_M"
    api_base_url: str = "https://api.groq.com/openai/v1"
    api_key: str = ""
    api_model: str = "llama-3.1-8b-instant"

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    chunk_target_tokens: int = 320   # bge-small max is 512; leave headroom
    chunk_overlap_sentences: int = 1
    top_k: int = 5
    min_similarity: float = 0.45     # cosine similarity gate; tune on eval set

    storage_dir: str = "storage"
    max_upload_mb: int = 25

    # --- Phase 2: auth ---
    jwt_secret: str = ""             # REQUIRED, >= 32 chars. Put it in backend/.env, never in git.
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 120

    # --- Phase 3: concept extraction ---
    concept_batch_chunks: int = 3          # chunks per LLM call (3 x ~320 tokens fits a 4K context)
    concept_max_chunks: int = 60           # only the first N chunks of a document are processed
    concept_max_per_doc: int = 40
    concept_merge_similarity: float = 0.92 # name-embedding cosine needed to auto-merge duplicates
    prereq_candidates: int = 8             # nearest concepts shown to the LLM per prerequisite query


@lru_cache
def get_settings() -> Settings:
    return Settings()
