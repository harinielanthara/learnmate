import re
from dataclasses import dataclass
from ..config import get_settings
from .embeddings import tokenizer

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Chunk:
    page_number: int
    text: str
    token_count: int


def clean_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)          # de-hyphenate line breaks
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)           # single newlines -> space
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _ntok(s: str) -> int:
    return len(tokenizer()(s, add_special_tokens=False)["input_ids"])


def chunk_page(text: str, page_number: int) -> list[Chunk]:
    """Sentence-packed chunks that never cross a page boundary (keeps citations exact).
    Trade-off: ideas spanning pages are split; acceptable for MVP."""
    s = get_settings()
    sents = [x.strip() for x in _SENT.split(clean_text(text)) if x.strip()]
    chunks, cur, cur_tok = [], [], 0
    for sent in sents:
        n = _ntok(sent)
        if n > s.chunk_target_tokens:  # pathological long "sentence": hard-split by words
            words, buf = sent.split(), []
            for w in words:
                buf.append(w)
                if _ntok(" ".join(buf)) >= s.chunk_target_tokens:
                    chunks.append(Chunk(page_number, " ".join(buf), _ntok(" ".join(buf))))
                    buf = []
            if buf:
                sent, n = " ".join(buf), _ntok(" ".join(buf))
            else:
                continue
        if cur and cur_tok + n > s.chunk_target_tokens:
            txt = " ".join(cur)
            chunks.append(Chunk(page_number, txt, cur_tok))
            cur = cur[-s.chunk_overlap_sentences:] if s.chunk_overlap_sentences else []
            cur_tok = sum(_ntok(c) for c in cur)
        cur.append(sent)
        cur_tok += n
    if cur:
        chunks.append(Chunk(page_number, " ".join(cur), cur_tok))
    return [c for c in chunks if c.token_count >= 20]  # drop tiny fragments (headers, page numbers)
