from fastapi import FastAPI
from .api import auth, concepts, conversations, documents, profile, qa, subjects

app = FastAPI(title="LearnMate API", version="0.3.0")
for r in (auth.router, subjects.router, profile.router, documents.router, qa.router,
          conversations.router, concepts.router):
    app.include_router(r)


@app.get("/health")
def health():
    return {"status": "ok"}
