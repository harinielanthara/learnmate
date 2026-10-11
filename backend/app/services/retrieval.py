from sqlalchemy import select, text
from sqlalchemy.orm import Session
from ..models import Document, DocumentChunk
from .embeddings import embed_query


def retrieve(db: Session, user_id: int, question: str, top_k: int = 5,
             document_ids: list[int] | None = None):
    """Cosine-distance kNN over this user's chunks only. Returns (chunk, filename, similarity)."""
    qv = embed_query(question)
    dist = DocumentChunk.embedding.cosine_distance(qv)
    stmt = (select(DocumentChunk, Document.filename, dist.label("dist"))
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.user_id == user_id))   # data isolation
    if document_ids:
        stmt = stmt.where(DocumentChunk.document_id.in_(document_ids))
    stmt = stmt.order_by(dist).limit(top_k)
    db.execute(text("SET LOCAL hnsw.ef_search = 60"))  # recall/latency knob
    return [(c, fn, 1.0 - float(d)) for c, fn, d in db.execute(stmt).all()]
