from pathlib import Path
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import delete, exists, select
from sqlalchemy.orm import Session
from ..config import get_settings
from ..db import get_db
from ..models import Concept, ConceptChunk, Document, Subject, User
from ..schemas import DocumentMove, DocumentOut
from ..services.ingestion import ALLOWED, ingest, save_upload, sha256_bytes
from .deps import get_current_user

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut)
def upload(file: UploadFile = File(...), subject_id: int | None = Form(None),
           db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    # plain `def` (not async): FastAPI runs it in a worker thread, so slow embedding
    # doesn't freeze the server for other requests.
    s = get_settings()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, "Only .pdf and .txt are supported.")
    data = file.file.read()
    if len(data) > s.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {s.max_upload_mb} MB.")
    if ext == ".pdf" and not data.startswith(b"%PDF"):
        raise HTTPException(400, "File is not a valid PDF.")
    if subject_id is not None and not db.scalar(
            select(Subject.id).where(Subject.id == subject_id, Subject.user_id == user.id)):
        raise HTTPException(404, "Subject not found")
    digest = sha256_bytes(data)
    dup = db.scalar(select(Document).where(Document.user_id == user.id, Document.sha256 == digest))
    if dup:
        if subject_id is not None and dup.subject_id != subject_id:
            dup.subject_id = subject_id   # re-uploading the same file into another subject = move it
            db.commit()
        return dup
    doc = Document(user_id=user.id, subject_id=subject_id, filename=(file.filename or "upload")[:255],
                   stored_path=save_upload(user.id, ext, data), sha256=digest)
    db.add(doc)
    db.commit()
    return ingest(db, doc, ext)


@router.get("", response_model=list[DocumentOut])
def list_documents(subject_id: int | None = None, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)):
    stmt = select(Document).where(Document.user_id == user.id)
    if subject_id is not None:
        stmt = stmt.where(Document.subject_id == subject_id)
    return db.scalars(stmt.order_by(Document.created_at.desc())).all()


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: int, db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)):
    doc = db.scalar(select(Document).where(Document.id == doc_id, Document.user_id == user.id))
    if not doc:
        raise HTTPException(404, "Not found")
    Path(doc.stored_path).unlink(missing_ok=True)
    db.delete(doc)   # chunks (and their concept links) removed by ON DELETE CASCADE
    db.commit()
    # drop AI-suggested, unverified concepts that no longer have any source passage
    db.execute(delete(Concept).where(
        Concept.user_id == user.id, Concept.source == "llm", Concept.verified.is_(False),
        ~exists().where(ConceptChunk.concept_id == Concept.id)))
    db.commit()


@router.patch("/{doc_id}", response_model=DocumentOut)
def move_document(doc_id: int, body: DocumentMove, db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)):
    """Assign a document to a subject (or unassign). Blocked once concepts were extracted from it."""
    doc = db.scalar(select(Document).where(Document.id == doc_id, Document.user_id == user.id))
    if not doc:
        raise HTTPException(404, "Not found")
    if doc.concept_status in ("running", "done"):
        raise HTTPException(409, "Concepts were already extracted from this document, so it can't be moved.")
    if body.subject_id is not None and not db.scalar(
            select(Subject.id).where(Subject.id == body.subject_id, Subject.user_id == user.id)):
        raise HTTPException(404, "Subject not found")
    doc.subject_id = body.subject_id
    db.commit()
    return doc
