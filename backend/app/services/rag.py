import re
from sqlalchemy.orm import Session
from ..config import get_settings
from ..llm.providers import get_llm
from ..schemas import AskResponse, Citation
from .retrieval import retrieve

REFUSAL = "I couldn't find enough information in your uploaded material to answer this reliably."
NOT_FOUND = "NOT_IN_SOURCES"   # sentinel the model emits when the sources don't help
GENERAL_LABEL = ("💡 *This isn't from your uploaded material. It's from my general knowledge, "
                 "so double-check important details.*\n\n")

GROUNDED_SYSTEM = f"""You are LearnMate, a friendly, patient tutor.
You are given numbered SOURCES from the student's own study material. Use them as your only facts.

How to answer:
- Explain in your OWN words, the way a good teacher would. Do NOT copy sentences or phrases from the sources.
- Sound natural and conversational: give a direct answer first, then a short explanation. Use a short list only if it really helps.
- Put citations like [1] or [2][3] at the end of sentences that use a source.
- Keep it concise (about 150 words or fewer) unless the question needs more.
- Only answer from the sources if they actually contain what is needed. A source merely mentioning the topic is NOT enough.
  If they don't contain the answer, reply with exactly: {NOT_FOUND}
- Never invent facts that are not in the sources.

Style example (do not reuse its content):
Source [1]: "Mitochondria: organelle; produces ATP via oxidative phosphorylation."
Good answer: "Mitochondria are the cell's power plants, because they make most of the cell's ATP energy [1]."
"""

GENERAL_SYSTEM = (
    "You are LearnMate, a friendly, patient tutor. Answer clearly and naturally, like a good teacher: "
    "a direct answer first, then a short explanation, and a simple example if it helps. "
    "Keep it under about 200 words. If you are unsure about something, say so instead of guessing."
)


def _used(out: str, n: int) -> set[int]:
    return {int(x) for x in re.findall(r"\[(\d+)\]", out) if 1 <= int(x) <= n}


def _words(t: str) -> list[str]:
    return re.findall(r"\w+", re.sub(r"\[\d+\]", "", t).lower())


def _copy_ratio(answer: str, sources: list[str], n: int = 6) -> float:
    """Share of the answer's 6-word sequences that appear verbatim in the sources (0..1)."""
    aw = _words(answer)
    if len(aw) < n + 2:
        return 0.0
    grams = {tuple(aw[i:i + n]) for i in range(len(aw) - n + 1)}
    src = set()
    for s in sources:
        sw = _words(s)
        src.update(tuple(sw[i:i + n]) for i in range(len(sw) - n + 1))
    return len(grams & src) / len(grams)


def general_answer(question: str) -> AskResponse:
    """Answer from the model's own knowledge. Always labelled, never cited."""
    llm = get_llm()
    out = llm.generate([{"role": "system", "content": GENERAL_SYSTEM},
                        {"role": "user", "content": question}],
                       temperature=0.4, max_tokens=500)
    return AskResponse(answer=GENERAL_LABEL + out, grounded=False, citations=[],
                       provider=llm.name, source="general")


def answer_question(db: Session, user_id: int, question: str, top_k: int,
                    document_ids: list[int] | None, mode: str = "auto") -> AskResponse:
    """mode: 'auto' = material first, general knowledge if the material doesn't help;
             'material' = strict, refuse if the material doesn't help (use this for evaluation)."""
    s = get_settings()
    hits = retrieve(db, user_id, question, top_k, document_ids)
    llm = get_llm()

    def fallback() -> AskResponse:
        if mode == "material":
            return AskResponse(answer=REFUSAL, grounded=False, citations=[],
                               provider=llm.name, source="none")
        return general_answer(question)

    # Gate 1: retrieval confidence
    if not hits or hits[0][2] < s.min_similarity:
        return fallback()

    ctx = "\n\n".join(f"[{i}] (p.{c.page_number}, {fn})\n{c.text}"
                      for i, (c, fn, _) in enumerate(hits, 1))
    src_texts = [c.text for c, _, _ in hits]
    messages = [
        {"role": "system", "content": GROUNDED_SYSTEM},
        {"role": "user", "content": f"SOURCES:\n{ctx}\n\nQuestion: {question}\nAnswer:"},
    ]
    out = llm.generate(messages, temperature=0.3, max_tokens=500)
    if NOT_FOUND in out or REFUSAL.lower()[:40] in out.lower():
        return fallback()

    n = len(hits)
    used, copied = _used(out, n), _copy_ratio(out, src_texts)
    if not used or copied > 0.5:   # forgot citations, or just pasted the source text: one retry
        fixes = []
        if not used:
            fixes.append("add source numbers like [1] after the sentences that use a source")
        if copied > 0.5:
            fixes.append("rewrite it in your own words, like a teacher explaining to a student, "
                         "without reusing phrases from the sources")
        out2 = llm.generate(messages + [
            {"role": "assistant", "content": out},
            {"role": "user", "content": "Revise your answer: " + "; and ".join(fixes) + "."},
        ], temperature=0.3, max_tokens=500)
        if NOT_FOUND not in out2:
            used2, copied2 = _used(out2, n), _copy_ratio(out2, src_texts)
            if (used2 or not used) and copied2 <= copied:   # accept only if no worse
                out, used = out2, used2

    grounded = bool(used)
    if not grounded:   # still uncited: show answer + closest passages, clearly flagged
        used = set(range(1, min(3, n) + 1))
        out += "\n\n⚠️ The model did not cite sources; the closest retrieved passages are shown below."

    cites = [Citation(ref=i, chunk_id=c.id, document_id=c.document_id, filename=fn,
                      page_number=c.page_number, similarity=round(sim, 3), text=c.text)
             for i, (c, fn, sim) in enumerate(hits, 1) if i in used]
    return AskResponse(answer=out, grounded=grounded, citations=cites,
                       provider=llm.name, source="material")
