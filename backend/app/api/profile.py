from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import Concept, Document, DocumentChunk, User
from ..schemas import ProfileOut
from .deps import get_current_user
from .subjects import subjects_with_counts

router = APIRouter(prefix="/profile", tags=["profile"])


@router.get("", response_model=ProfileOut)
def get_profile(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Student profile (so far: who they are + what they've uploaded).
    Mastery/quiz history get added here in later phases. Chat history is NOT part of it."""
    subs = subjects_with_counts(db, user.id)
    docs = db.scalar(select(func.count(Document.id)).where(Document.user_id == user.id)) or 0
    ready = db.scalar(select(func.count(Document.id)).where(
        Document.user_id == user.id, Document.status == "ready")) or 0
    chunks = db.scalar(select(func.count(DocumentChunk.id)).where(
        DocumentChunk.user_id == user.id)) or 0
    n_concepts = db.scalar(select(func.count(Concept.id)).where(Concept.user_id == user.id)) or 0
    n_verified = db.scalar(select(func.count(Concept.id)).where(
        Concept.user_id == user.id, Concept.verified.is_(True))) or 0
    return ProfileOut(user=user, subject_count=len(subs), document_count=docs,
                      ready_document_count=ready, chunk_count=chunks,
                      concept_count=n_concepts, verified_concept_count=n_verified, subjects=subs)
