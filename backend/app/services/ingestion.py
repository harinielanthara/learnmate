import hashlib
import uuid
from pathlib import Path
from pypdf import PdfReader
from sqlalchemy.orm import Session
from ..config import get_settings
from ..models import Document, DocumentChunk
from .chunking import chunk_page
from .embeddings import embed_passages

ALLOWED = {".pdf", ".txt"}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def save_upload(user_id: int, ext: str, data: bytes) -> str:
    d = Path(get_settings().storage_dir) / str(user_id)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{uuid.uuid4().hex}{ext}"   # never trust the client filename for paths
    p.write_bytes(data)
    return str(p)


def extract_pages(path: str, ext: str) -> list[tuple[int, str]]:
    if ext == ".pdf":
        reader = PdfReader(path)
        return [(i + 1, (pg.extract_text() or "")) for i, pg in enumerate(reader.pages)]
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    # TXT has no pages: pseudo-pages of ~3000 chars so citations still point somewhere
    return [(i // 3000 + 1, text[i:i + 3000]) for i in range(0, len(text), 3000)] or [(1, "")]


def ingest(db: Session, doc: Document, ext: str) -> Document:
    try:
        pages = extract_pages(doc.stored_path, ext)
        doc.page_count = len(pages)
        chunks = []
        for pno, txt in pages:
            chunks.extend(chunk_page(txt, pno))
        if not chunks:
            raise ValueError("No extractable text (scanned PDF? OCR is out of MVP scope).")
        vecs = embed_passages([c.text for c in chunks])
        db.add_all([
            DocumentChunk(document_id=doc.id, user_id=doc.user_id, chunk_index=i,
                          page_number=c.page_number, text=c.text,
                          token_count=c.token_count, embedding=v)
            for i, (c, v) in enumerate(zip(chunks, vecs))
        ])
        doc.chunk_count, doc.status = len(chunks), "ready"
    except Exception as e:  # noqa: BLE001
        doc.status, doc.error = "failed", str(e)[:500]
    db.commit()
    return doc
