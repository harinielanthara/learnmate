# LearnMate — Phase 1 (ingest → embed → pgvector → grounded RAG Q&A)

## Run (Windows, VSCode terminal, env `learnmode`)
```powershell
# 0. Ollama (native, uses GPU):  https://ollama.com  then:
ollama pull qwen2.5:3b-instruct-q4_K_M

# 1. Database
copy .env.example backend\.env
docker compose up -d db

# 2. Backend
conda activate learnmode
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
cd backend
alembic revision --autogenerate -m "init"
#   -> open the new file in alembic/versions and add `import pgvector.sqlalchemy` at the top
alembic upgrade head
uvicorn app.main:app --reload --port 8000

# 3. Frontend (new terminal)
cd frontend
streamlit run app.py
```
Docs: http://localhost:8000/docs

## Smoke test
Upload a PDF, ask something in it (expect cited answer), then ask something unrelated
(expect the refusal sentence).

## VRAM budget (RTX 4060, 8 GB)
Qwen2.5-3B Q4 ≈ 2 GB weights + ≈0.5 GB KV cache @ 4K ctx → ~2.5–3 GB. Embeddings run on CPU (0 GB).
You have room for a 7B Q4_K_M (~4.7 GB + cache) but it is tight with a browser/desktop compositor; keep it experimental.
