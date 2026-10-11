import re
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import ChatMessage, Conversation, Document, Subject, User
from ..schemas import AskRequest, AskResponse
from ..services.rag import REFUSAL, answer_question, general_answer
from .deps import get_current_user

router = APIRouter(prefix="/qa", tags=["qa"])


def _title(question: str) -> str:
    t = re.sub(r"\s+", " ", question).strip()
    return t if len(t) <= 60 else t[:57].rstrip() + "..."


@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    conv = None
    if req.conversation_id is not None:
        conv = db.scalar(select(Conversation).where(Conversation.id == req.conversation_id,
                                                    Conversation.user_id == user.id))
        if not conv:
            raise HTTPException(404, "Conversation not found")

    doc_ids = req.document_ids
    no_material = False
    if req.subject_id is not None:
        if not db.scalar(select(Subject.id).where(Subject.id == req.subject_id,
                                                  Subject.user_id == user.id)):
            raise HTTPException(404, "Subject not found")
        in_subject = set(db.scalars(select(Document.id).where(
            Document.user_id == user.id, Document.subject_id == req.subject_id,
            Document.status == "ready")))
        doc_ids = [d for d in in_subject if not req.document_ids or d in set(req.document_ids)]
        # IMPORTANT: an empty list would mean "no filter" downstream, so handle it here
        no_material = not doc_ids
    try:
        if req.mode == "general":
            resp = general_answer(req.question)
        elif no_material:
            resp = (AskResponse(answer=REFUSAL, grounded=False, citations=[], provider="none",
                                source="none")
                    if req.mode == "material" else general_answer(req.question))
        else:
            resp = answer_question(db, user.id, req.question, req.top_k, doc_ids, req.mode)
    except Exception as e:  # noqa: BLE001  (LLM down, etc.)
        raise HTTPException(502, f"LLM/provider error: {e}") from e

    if conv is None:   # first message of a new chat -> create the saved conversation
        conv = Conversation(user_id=user.id, subject_id=req.subject_id, title=_title(req.question))
        db.add(conv)
        db.flush()
    conv.updated_at = datetime.now(timezone.utc)
    db.add_all([
        ChatMessage(user_id=user.id, conversation_id=conv.id, subject_id=req.subject_id,
                    role="user", content=req.question),
        ChatMessage(user_id=user.id, conversation_id=conv.id, subject_id=req.subject_id,
                    role="assistant", content=resp.answer,
                    citations=[c.model_dump() for c in resp.citations] or None),
    ])
    db.commit()
    resp.conversation_id = conv.id
    return resp
