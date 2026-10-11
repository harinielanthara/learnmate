"""Concept extraction + prerequisite inference.

Pipeline (all heuristic, all LLM output is validated and stored as source='llm', verified=False):
  chunks -> batches -> LLM JSON -> normalize + dedupe -> concept<->chunk links
         -> windows of concepts -> LLM JSON edges -> validate, order-check, cycle-check -> edges
"""
import json
import re
from collections import defaultdict
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..llm.base import LLMProvider
from ..models import Concept, ConceptChunk, ConceptPrerequisite, Document, DocumentChunk
from .embeddings import embed_passages

GENERIC = {"introduction", "example", "examples", "figure", "table", "chapter", "summary",
           "overview", "conclusion", "exercise", "exercises", "page", "section", "definition",
           "note", "notes", "references", "contents", "topic", "concept", "answer", "question"}

EXTRACT_SYSTEM = """You extract the key study concepts from a student's study material.
Return JSON only, in exactly this shape:
{"concepts": [{"name": "...", "description": "..."}]}
Rules:
- 3 to 8 concepts. Each name is a short noun phrase (1-5 words), singular, a term a student must learn.
- description: ONE sentence in your own words.
- Skip people, institutions, contact details, headers, and generic words like "introduction" or "example".
- Only include concepts that actually appear in the text."""

PREREQ_SYSTEM = """You build a prerequisite graph for a student's study material.
Given a list of concepts, list the prerequisite relationships between them:
A is a prerequisite of B if a student must understand A before they can understand B.
Return JSON only, in exactly this shape:
{"edges": [{"concept": "B", "prerequisite": "A", "confidence": 0.8}]}
Rules:
- Use ONLY concept names from the list, written exactly as given.
- Include only clear, direct relationships. Fewer is better. No cycles.
- confidence is a number from 0 to 1."""


# ------------------------------------------------------------------ pure helpers
def normalize_name(name: str) -> str:
    n = name.lower().strip()
    n = re.sub(r"\([^)]*\)", " ", n)            # "Support Vector Machine (SVM)" -> "support vector machine"
    n = re.sub(r"[-_]+", " ", n)
    n = re.sub(r"[^a-z0-9+#\s]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    n = re.sub(r"^(the|a|an)\s+", "", n)
    words = n.split()
    if words:
        w = words[-1]
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("s") and len(w) > 3 and not w.endswith(("ss", "us", "is", "as", "os")):
            w = w[:-1]
        words[-1] = w
    return " ".join(words)


def valid_name(name: str) -> bool:
    n = normalize_name(name)
    return (2 <= len(name.strip()) <= 60 and len(n) >= 2 and n not in GENERIC
            and re.search(r"[a-z]", n) is not None)


def parse_json_obj(text: str) -> dict | None:
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)   # tolerate ```json fences / chatter
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def similar_names(a_norm: str, b_norm: str, cosine: float) -> bool:
    """Embedding similarity alone is unsafe for short terms (bge-small scores most pairs high),
    so it must be backed by word overlap. 'linear regression' != 'logistic regression'."""
    ta, tb = set(a_norm.split()), set(b_norm.split())
    if min(len(ta), len(tb)) < 2:     # single-word terms are too ambiguous to auto-merge
        return False
    jaccard = len(ta & tb) / len(ta | tb)
    subset = ta <= tb or tb <= ta
    return cosine >= 0.90 and (jaccard >= 0.6 or subset)


def dot(a: list[float], b: list[float]) -> float:   # vectors are L2-normalized -> dot = cosine
    return sum(x * y for x, y in zip(a, b))


def make_batches(chunks: list, max_tokens: int = 900) -> list[list]:
    batches, cur, tok = [], [], 0
    for c in chunks:
        if cur and (tok + c.token_count > max_tokens or c.document_id != cur[-1].document_id):
            batches.append(cur)
            cur, tok = [], 0
        cur.append(c)
        tok += c.token_count
    if cur:
        batches.append(cur)
    return batches


def extract_from_text(llm: LLMProvider, text: str) -> list[tuple[str, str]] | None:
    """Returns [(name, description)], or None if the model never produced valid JSON."""
    for _ in range(2):
        out = llm.generate([{"role": "system", "content": EXTRACT_SYSTEM},
                            {"role": "user", "content": f"STUDY MATERIAL:\n{text}\n\nJSON:"}],
                           temperature=0.1, max_tokens=500, json_mode=True)
        obj = parse_json_obj(out)
        items = obj.get("concepts") if obj else None
        if isinstance(items, list):
            res = []
            for it in items[:10]:
                if isinstance(it, dict) and isinstance(it.get("name"), str) and valid_name(it["name"]):
                    d = it.get("description")
                    res.append((it["name"].strip().rstrip("."),
                                d.strip()[:300] if isinstance(d, str) else ""))
            return res
    return None


def parse_edges(text: str, name_to_id: dict[str, int]) -> list[tuple[int, int, float]]:
    """name_to_id maps NORMALIZED names -> concept id. Returns [(concept_id, prereq_id, confidence)]."""
    obj = parse_json_obj(text)
    items = obj.get("edges") if obj else None
    out = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        c, p = it.get("concept"), it.get("prerequisite")
        if not isinstance(c, str) or not isinstance(p, str):
            continue
        cid, pid = name_to_id.get(normalize_name(c)), name_to_id.get(normalize_name(p))
        if cid is None or pid is None or cid == pid:
            continue
        try:
            conf = max(0.0, min(1.0, float(it.get("confidence", 0.6))))
        except (TypeError, ValueError):
            conf = 0.6
        out.append((cid, pid, conf))
    return out


def order_adjust(conf: float, pos_concept, pos_prereq) -> float:
    """Heuristic: within ONE document a prerequisite usually appears before the concept that needs it.
    If it appears later, trust the edge less."""
    if pos_concept and pos_prereq and pos_concept[0] == pos_prereq[0] and pos_prereq[1] > pos_concept[1]:
        return conf * 0.6
    return conf


def reaches(requires: dict[int, set[int]], start: int, target: int) -> bool:
    """True if `target` is reachable from `start` by following 'requires' edges."""
    seen, stack = set(), [start]
    while stack:
        x = stack.pop()
        if x == target:
            return True
        if x in seen:
            continue
        seen.add(x)
        stack.extend(requires.get(x, ()))
    return False


def would_cycle(requires: dict[int, set[int]], concept_id: int, prereq_id: int) -> bool:
    """Adding 'concept_id requires prereq_id' makes a cycle iff prereq_id already (transitively) requires concept_id."""
    return reaches(requires, prereq_id, concept_id)


# ------------------------------------------------------------------ DB helpers
def load_requires(db: Session, subject_id: int) -> dict[int, set[int]]:
    """Adjacency of live edges (not rejected, both ends active) in one subject."""
    active = {i for (i,) in db.execute(select(Concept.id).where(
        Concept.subject_id == subject_id, Concept.active.is_(True)))}
    req: dict[int, set[int]] = defaultdict(set)
    if not active:
        return req
    for e in db.scalars(select(ConceptPrerequisite).where(
            ConceptPrerequisite.concept_id.in_(active), ConceptPrerequisite.rejected.is_(False))):
        if e.prerequisite_id in active:
            req[e.concept_id].add(e.prerequisite_id)
    return req


@dataclass
class ExtractStats:
    chunks_processed: int = 0
    batches: int = 0
    batches_failed: int = 0
    new_concepts: int = 0
    reused_concepts: int = 0
    new_links: int = 0
    new_prerequisites: int = 0


def _resolve_existing(by_norm: dict[str, Concept], by_id: dict[int, Concept], norm: str) -> Concept | None | str:
    """Exact-name lookup honoring tombstones. Returns Concept, None (unknown), or 'SKIP' (user deleted it)."""
    row = by_norm.get(norm)
    if row is None:
        return None
    for _ in range(5):                       # follow merge chain
        if row.active:
            return row
        if row.merged_into_id and row.merged_into_id in by_id:
            row = by_id[row.merged_into_id]
            continue
        return "SKIP"
    return "SKIP"


def extract_subject_concepts(db: Session, llm: LLMProvider, user_id: int, subject_id: int,
                             document_ids: list[int] | None, max_chunks: int) -> ExtractStats:
    st = ExtractStats()
    q = (select(DocumentChunk).join(Document, Document.id == DocumentChunk.document_id)
         .where(Document.user_id == user_id, Document.subject_id == subject_id,
                Document.status == "ready"))
    if document_ids:
        q = q.where(DocumentChunk.document_id.in_(document_ids))
    chunks = list(db.scalars(q.order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
                             .limit(max_chunks)))
    st.chunks_processed = len(chunks)
    if not chunks:
        return st

    all_concepts = list(db.scalars(select(Concept).where(Concept.subject_id == subject_id,
                                                         Concept.user_id == user_id)))
    by_norm = {c.name_normalized: c for c in all_concepts}
    by_id = {c.id: c for c in all_concepts}
    emb: dict[int, list[float]] = {}
    fallback_links: dict[int, set[int]] = defaultdict(set)   # concept_id -> chunk ids of its batch
    new_ids: set[int] = set()

    def vec(c: Concept) -> list[float]:
        if c.id not in emb:
            emb[c.id] = embed_passages([c.name_normalized])[0]
        return emb[c.id]

    for batch in make_batches(chunks):
        st.batches += 1
        found = extract_from_text(llm, "\n\n".join(c.text for c in batch))
        if found is None:
            st.batches_failed += 1
            continue
        for name, desc in found:
            norm = normalize_name(name)
            hit = _resolve_existing(by_norm, by_id, norm)
            if hit == "SKIP":
                continue
            target = hit if isinstance(hit, Concept) else None
            if target is None:   # fuzzy match against live concepts (embedding + word overlap)
                nv = embed_passages([norm])[0]
                for c in by_id.values():
                    if c.active and similar_names(norm, c.name_normalized, dot(nv, vec(c))):
                        target = c
                        break
            if target is None:
                target = Concept(user_id=user_id, subject_id=subject_id, name=name[:120],
                                 name_normalized=norm[:120], description=desc or None,
                                 source="llm", verified=False)
                db.add(target)
                db.flush()
                by_norm[norm], by_id[target.id] = target, target
                new_ids.add(target.id)
                st.new_concepts += 1
            else:
                st.reused_concepts += 1
                if not target.description and desc and not target.verified:
                    target.description = desc
            fallback_links[target.id].update(c.id for c in batch)
        db.commit()

    # --- concept <-> chunk links: name occurs in the passage, else fall back to the batch it came from
    live = [c for c in by_id.values() if c.active]
    existing = {(cid, kid) for cid, kid in db.execute(
        select(ConceptChunk.concept_id, ConceptChunk.chunk_id).where(
            ConceptChunk.concept_id.in_([c.id for c in live] or [0])))}
    norm_text = {k.id: re.sub(r"[-_]+", " ", k.text.lower()) for k in chunks}
    for c in live:
        pat = re.compile(r"\b" + re.escape(c.name_normalized) + r"s?\b")
        hits = {k.id for k in chunks if pat.search(norm_text[k.id])}
        if not hits and c.id in new_ids:
            hits = fallback_links.get(c.id, set())
        for kid in hits:
            if (c.id, kid) not in existing:
                db.add(ConceptChunk(concept_id=c.id, chunk_id=kid))
                existing.add((c.id, kid))
                st.new_links += 1
    db.commit()

    st.new_prerequisites = infer_prerequisites(db, llm, subject_id, new_ids)
    return st


def infer_prerequisites(db: Session, llm: LLMProvider, subject_id: int, new_ids: set[int]) -> int:
    live = list(db.scalars(select(Concept).where(Concept.subject_id == subject_id,
                                                 Concept.active.is_(True))))
    if len(live) < 2 or not new_ids:
        return 0
    # first position of each concept: (document_id, chunk_index) of its earliest linked passage
    pos: dict[int, tuple[int, int]] = {}
    for cid, doc_id, idx in db.execute(
            select(ConceptChunk.concept_id, DocumentChunk.document_id, DocumentChunk.chunk_index)
            .join(DocumentChunk, DocumentChunk.id == ConceptChunk.chunk_id)
            .where(ConceptChunk.concept_id.in_([c.id for c in live]))):
        if cid not in pos or (doc_id, idx) < pos[cid]:
            pos[cid] = (doc_id, idx)
    live.sort(key=lambda c: (pos.get(c.id, (10**9, 0)), c.id))

    size, stride = 20, 12
    windows = [live] if len(live) <= size else [live[i:i + size] for i in range(0, len(live) - size + stride, stride)]
    existing = {(e.concept_id, e.prerequisite_id) for e in db.scalars(select(ConceptPrerequisite).where(
        ConceptPrerequisite.concept_id.in_([c.id for c in live])))}
    requires = load_requires(db, subject_id)
    added = 0
    for w in windows:
        if not any(c.id in new_ids for c in w):
            continue
        listing = "\n".join(f"- {c.name}" + (f": {c.description}" if c.description else "") for c in w)
        out = llm.generate([{"role": "system", "content": PREREQ_SYSTEM},
                            {"role": "user", "content": f"CONCEPTS:\n{listing}\n\nJSON:"}],
                           temperature=0.1, max_tokens=700, json_mode=True)
        edges = parse_edges(out, {c.name_normalized: c.id for c in w})
        edges = sorted(((c, p, order_adjust(cf, pos.get(c), pos.get(p))) for c, p, cf in edges),
                       key=lambda e: -e[2])[: 2 * len(w)]
        for cid, pid, conf in edges:
            if conf < 0.4 or (cid, pid) in existing or would_cycle(requires, cid, pid):
                continue
            db.add(ConceptPrerequisite(concept_id=cid, prerequisite_id=pid, source="llm",
                                       verified=False, confidence=round(conf, 2)))
            existing.add((cid, pid))
            requires[cid].add(pid)
            added += 1
        db.commit()
    return added
