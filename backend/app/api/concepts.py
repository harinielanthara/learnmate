from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from ..db import get_db
from ..llm.providers import get_llm
from ..models import (Concept, ConceptChunk, ConceptPrerequisite, Document, DocumentChunk,
                      Subject, User)
from ..schemas import (ConceptCreate, ConceptOut, ConceptUpdate, ExtractRequest, ExtractResult,
                       MergeRequest, PrereqCreate, PrereqOut, PrereqUpdate, SourceChunkOut)
from ..services.concepts import (extract_subject_concepts, load_requires, normalize_name,
                                 valid_name, would_cycle)
from .deps import get_current_user

router = APIRouter(prefix="/concepts", tags=["concepts"])


# ------------------------------------------------------------------ helpers
def _subject(db: Session, subject_id: int, user: User) -> Subject:
    s = db.scalar(select(Subject).where(Subject.id == subject_id, Subject.user_id == user.id))
    if not s:
        raise HTTPException(404, "Subject not found")
    return s


def _concept(db: Session, cid: int, user: User, active_only: bool = True) -> Concept:
    c = db.scalar(select(Concept).where(Concept.id == cid, Concept.user_id == user.id))
    if not c or (active_only and not c.active):
        raise HTTPException(404, "Concept not found")
    return c


def build_concept_list(db: Session, user_id: int, subject_id: int) -> list[ConceptOut]:
    concepts = list(db.scalars(select(Concept).where(
        Concept.user_id == user_id, Concept.subject_id == subject_id, Concept.active.is_(True))
        .order_by(Concept.name)))
    ids = [c.id for c in concepts]
    if not ids:
        return []
    chunk_counts = dict(db.execute(select(ConceptChunk.concept_id, func.count())
                                   .where(ConceptChunk.concept_id.in_(ids))
                                   .group_by(ConceptChunk.concept_id)).all())
    by_id = {c.id: c for c in concepts}
    prereqs, dependents = defaultdict(list), defaultdict(int)
    for e in db.scalars(select(ConceptPrerequisite).where(
            ConceptPrerequisite.concept_id.in_(ids), ConceptPrerequisite.rejected.is_(False))):
        if e.prerequisite_id in by_id:
            p = by_id[e.prerequisite_id]
            prereqs[e.concept_id].append(PrereqOut(id=p.id, name=p.name, source=e.source,
                                                   verified=e.verified, confidence=e.confidence))
            dependents[e.prerequisite_id] += 1
    return [ConceptOut(id=c.id, subject_id=c.subject_id, name=c.name, description=c.description,
                       source=c.source, verified=c.verified, chunk_count=chunk_counts.get(c.id, 0),
                       dependent_count=dependents.get(c.id, 0),
                       prerequisites=sorted(prereqs.get(c.id, []), key=lambda p: p.name))
            for c in concepts]


def _one(db: Session, user_id: int, c: Concept) -> ConceptOut:
    return next(x for x in build_concept_list(db, user_id, c.subject_id) if x.id == c.id)


# ------------------------------------------------------------------ extraction
@router.post("/extract", response_model=ExtractResult)
def extract(req: ExtractRequest, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Runs the local LLM over this subject's passages. Synchronous: expect ~1-3 min per 40 passages."""
    _subject(db, req.subject_id, user)
    if req.document_ids:
        ok = set(db.scalars(select(Document.id).where(
            Document.user_id == user.id, Document.subject_id == req.subject_id,
            Document.id.in_(req.document_ids))))
        if ok != set(req.document_ids):
            raise HTTPException(404, "One or more documents are not in this subject")
    llm = get_llm()
    try:
        st = extract_subject_concepts(db, llm, user.id, req.subject_id, req.document_ids, req.max_chunks)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"LLM/provider error: {e}") from e
    return ExtractResult(provider=llm.name, **st.__dict__)


# ------------------------------------------------------------------ read
@router.get("", response_model=list[ConceptOut])
def list_concepts(subject_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _subject(db, subject_id, user)
    return build_concept_list(db, user.id, subject_id)


@router.get("/{cid}/chunks", response_model=list[SourceChunkOut])
def concept_chunks(cid: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """The passages a concept was extracted from, so you can verify it."""
    _concept(db, cid, user)
    rows = db.execute(
        select(DocumentChunk, Document.filename)
        .join(ConceptChunk, ConceptChunk.chunk_id == DocumentChunk.id)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(ConceptChunk.concept_id == cid, DocumentChunk.user_id == user.id)
        .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index).limit(5)).all()
    return [SourceChunkOut(chunk_id=k.id, document_id=k.document_id, filename=fn,
                           page_number=k.page_number, text=k.text[:500]) for k, fn in rows]


# ------------------------------------------------------------------ manual corrections
@router.post("", response_model=ConceptOut, status_code=201)
def create_concept(body: ConceptCreate, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _subject(db, body.subject_id, user)
    name = body.name.strip()
    if not valid_name(name):
        raise HTTPException(400, "That is not a usable concept name.")
    norm = normalize_name(name)
    row = db.scalar(select(Concept).where(Concept.user_id == user.id, Concept.subject_id == body.subject_id,
                                          Concept.name_normalized == norm))
    if row and row.active:
        raise HTTPException(409, "That concept already exists.")
    if row:   # reactivate a deleted/merged tombstone
        row.active, row.merged_into_id = True, None
        row.name, row.description = name, body.description
        row.source, row.verified = "manual", True
    else:
        row = Concept(user_id=user.id, subject_id=body.subject_id, name=name, name_normalized=norm,
                      description=body.description, source="manual", verified=True)
        db.add(row)
    db.commit()
    return _one(db, user.id, row)


@router.patch("/{cid}", response_model=ConceptOut)
def update_concept(cid: int, body: ConceptUpdate, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)):
    c = _concept(db, cid, user)
    if body.name is not None:
        name = body.name.strip()
        if not valid_name(name):
            raise HTTPException(400, "That is not a usable concept name.")
        norm = normalize_name(name)
        clash = db.scalar(select(Concept.id).where(
            Concept.user_id == user.id, Concept.subject_id == c.subject_id,
            Concept.name_normalized == norm, Concept.id != c.id))
        if clash:
            raise HTTPException(409, "Another (or a deleted/merged) concept already uses that name.")
        c.name, c.name_normalized = name, norm
    if body.description is not None:
        c.description = body.description.strip() or None
    if body.verified is not None:
        c.verified = body.verified
    db.commit()
    return _one(db, user.id, c)


@router.delete("/{cid}", status_code=204)
def delete_concept(cid: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    c = _concept(db, cid, user)
    c.active = False    # tombstone: re-running extraction will not bring it back
    db.commit()


@router.post("/{cid}/merge", response_model=ConceptOut)
def merge_concept(cid: int, body: MergeRequest, db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)):
    """Merge concept `cid` INTO `into_id` (duplicates, synonyms, acronyms)."""
    a, b = _concept(db, cid, user), _concept(db, body.into_id, user)
    if a.id == b.id or a.subject_id != b.subject_id:
        raise HTTPException(400, "Choose a different concept from the same subject.")
    have = {k for (k,) in db.execute(select(ConceptChunk.chunk_id).where(ConceptChunk.concept_id == b.id))}
    for (k,) in db.execute(select(ConceptChunk.chunk_id).where(ConceptChunk.concept_id == a.id)).all():
        if k not in have:
            db.add(ConceptChunk(concept_id=b.id, chunk_id=k))
    for link in db.scalars(select(ConceptChunk).where(ConceptChunk.concept_id == a.id)):
        db.delete(link)
    edges_a = list(db.scalars(select(ConceptPrerequisite).where(
        (ConceptPrerequisite.concept_id == a.id) | (ConceptPrerequisite.prerequisite_id == a.id))))
    a.active, a.merged_into_id = False, b.id      # tombstone first so cycle checks ignore A
    db.flush()
    requires = load_requires(db, a.subject_id)
    existing = {(e.concept_id, e.prerequisite_id) for e in db.scalars(select(ConceptPrerequisite).where(
        (ConceptPrerequisite.concept_id == b.id) | (ConceptPrerequisite.prerequisite_id == b.id)))}
    for e in edges_a:
        if not e.rejected:
            c_id = b.id if e.concept_id == a.id else e.concept_id
            p_id = b.id if e.prerequisite_id == a.id else e.prerequisite_id
            if (c_id != p_id and (c_id, p_id) not in existing and not would_cycle(requires, c_id, p_id)):
                db.add(ConceptPrerequisite(concept_id=c_id, prerequisite_id=p_id, source=e.source,
                                           verified=e.verified, confidence=e.confidence))
                existing.add((c_id, p_id))
                requires[c_id].add(p_id)
        db.delete(e)
    db.commit()
    return _one(db, user.id, b)


@router.post("/{cid}/prerequisites", response_model=ConceptOut, status_code=201)
def add_prerequisite(cid: int, body: PrereqCreate, db: Session = Depends(get_db),
                     user: User = Depends(get_current_user)):
    """Manual edge: concept `cid` REQUIRES `prerequisite_id`."""
    c, p = _concept(db, cid, user), _concept(db, body.prerequisite_id, user)
    if c.id == p.id:
        raise HTTPException(400, "A concept cannot be its own prerequisite.")
    if c.subject_id != p.subject_id:
        raise HTTPException(400, "Both concepts must be in the same subject.")
    if would_cycle(load_requires(db, c.subject_id), c.id, p.id):
        raise HTTPException(409, "That would create a circular prerequisite chain.")
    e = db.get(ConceptPrerequisite, (c.id, p.id))
    if e:   # also revives an edge the user previously removed
        e.rejected, e.source, e.verified, e.confidence = False, "manual", True, 1.0
    else:
        db.add(ConceptPrerequisite(concept_id=c.id, prerequisite_id=p.id, source="manual",
                                   verified=True, confidence=1.0))
    db.commit()
    return _one(db, user.id, c)


@router.patch("/{cid}/prerequisites/{pid}", response_model=ConceptOut)
def verify_prerequisite(cid: int, pid: int, body: PrereqUpdate, db: Session = Depends(get_db),
                        user: User = Depends(get_current_user)):
    c = _concept(db, cid, user)
    e = db.get(ConceptPrerequisite, (cid, pid))
    if not e or e.rejected:
        raise HTTPException(404, "Prerequisite link not found")
    e.verified = body.verified    # `source` keeps recording where it came from (llm/manual)
    db.commit()
    return _one(db, user.id, c)


@router.delete("/{cid}/prerequisites/{pid}", response_model=ConceptOut)
def remove_prerequisite(cid: int, pid: int, db: Session = Depends(get_db),
                        user: User = Depends(get_current_user)):
    c = _concept(db, cid, user)
    e = db.get(ConceptPrerequisite, (cid, pid))
    if not e or e.rejected:
        raise HTTPException(404, "Prerequisite link not found")
    e.rejected = True   # tombstone, so re-extraction does not re-add it
    db.commit()
    return _one(db, user.id, c)
