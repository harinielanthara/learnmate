from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import Document, Subject, User
from ..schemas import SubjectCreate, SubjectOut
from .deps import get_current_user

router = APIRouter(prefix="/subjects", tags=["subjects"])


def subjects_with_counts(db: Session, user_id: int) -> list[SubjectOut]:
    rows = db.execute(
        select(Subject, func.count(Document.id))
        .outerjoin(Document, Document.subject_id == Subject.id)
        .where(Subject.user_id == user_id)
        .group_by(Subject.id).order_by(Subject.name)).all()
    return [SubjectOut(id=s.id, name=s.name, created_at=s.created_at, document_count=n)
            for s, n in rows]


@router.post("", response_model=SubjectOut, status_code=201)
def create_subject(body: SubjectCreate, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Subject name cannot be empty.")
    if db.scalar(select(Subject.id).where(Subject.user_id == user.id, Subject.name == name)):
        raise HTTPException(409, "You already have a subject with that name.")
    s = Subject(user_id=user.id, name=name)
    db.add(s)
    db.commit()
    return SubjectOut(id=s.id, name=s.name, created_at=s.created_at, document_count=0)


@router.get("", response_model=list[SubjectOut])
def list_subjects(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return subjects_with_counts(db, user.id)


@router.delete("/{subject_id}", status_code=204)
def delete_subject(subject_id: int, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)):
    s = db.scalar(select(Subject).where(Subject.id == subject_id, Subject.user_id == user.id))
    if not s:
        raise HTTPException(404, "Subject not found")
    db.delete(s)   # its documents stay (subject_id -> NULL via ON DELETE SET NULL)
    db.commit()
