import sys
from sqlalchemy import select
from app.config import get_settings
from app.db import SessionLocal
from app.models import Document
from app.services.retrieval import retrieve
from app.services.rag import SYSTEM
from app.llm.providers import get_llm

q = " ".join(sys.argv[1:]) or "where does harini learn?"
db, s = SessionLocal(), get_settings()
print("MIN_SIMILARITY =", s.min_similarity, "| model =", s.ollama_model)

for d in db.scalars(select(Document)):
    print(f"doc {d.id} {d.filename} status={d.status} pages={d.page_count} chunks={d.chunk_count} err={d.error}")

user_id = db.scalar(select(Document.user_id).limit(1))
hits = retrieve(db, user_id, q, 5)
print("\n--- TOP CHUNKS ---")
for i, (c, fn, sim) in enumerate(hits, 1):
    print(f"[{i}] sim={sim:.3f} page={c.page_number} tokens={c.token_count}")
    print("    ", c.text[:200].replace("\n", " "), "\n")

ctx = "\n\n".join(f"[{i}] (p.{c.page_number})\n{c.text}" for i, (c, fn, _) in enumerate(hits, 1))
out = get_llm().generate([
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {q}\nAnswer:"},
])
print("--- RAW LLM OUTPUT ---\n", out)