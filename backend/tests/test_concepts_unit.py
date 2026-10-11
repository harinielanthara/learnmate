"""Pure-logic tests (no database, no LLM).  Run from backend/:  python -m pytest tests -q
(install once: pip install pytest)"""
import sys, types, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
st = types.ModuleType("sentence_transformers"); st.SentenceTransformer = object
sys.modules.setdefault("sentence_transformers", st)   # avoid loading torch in unit tests

from types import SimpleNamespace as NS
from app.services.concepts import (extract_from_text, make_batches, normalize_name, order_adjust,
                                   parse_edges, parse_json_obj, similar_names, valid_name, would_cycle)


def test_normalize():
    assert normalize_name("Neural Networks") == normalize_name("neural network") == "neural network"
    assert normalize_name("Support Vector Machine (SVM)") == "support vector machine"
    assert normalize_name("The Loss-Function") == "loss function"
    assert normalize_name("Bias") == "bias" and normalize_name("Analysis") == "analysis"
    assert normalize_name("Policies") == "policy"


def test_valid_name():
    assert valid_name("Gradient Descent")
    assert not valid_name("Introduction") and not valid_name("A") and not valid_name("12345")
    assert not valid_name("x" * 80)


def test_parse_json_tolerates_fences_and_chatter():
    assert parse_json_obj('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_obj('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert parse_json_obj("not json") is None and parse_json_obj("[1,2]") is None


def test_similar_names_needs_word_overlap():
    assert similar_names("gradient descent", "gradient descent algorithm", 0.93)       # subset, 2+ words
    assert not similar_names("linear regression", "logistic regression", 0.95)        # high cosine alone is not enough
    assert not similar_names("regression", "linear regression", 0.95)                 # 1-word subset not allowed
    assert not similar_names("gradient descent", "gradient descent algorithm", 0.80)  # cosine too low


def test_batches_never_cross_documents_and_respect_budget():
    ch = [NS(document_id=1, token_count=400), NS(document_id=1, token_count=400),
          NS(document_id=1, token_count=400), NS(document_id=2, token_count=100)]
    b = make_batches(ch, 900)
    assert [len(x) for x in b] == [2, 1, 1] and b[2][0].document_id == 2


class Fake:
    def __init__(self, outs): self.outs = list(outs)
    def generate(self, *a, **k): return self.outs.pop(0)


def test_extract_retries_then_succeeds_and_filters():
    llm = Fake(["garbage", json.dumps({"concepts": [
        {"name": "Gradient Descent", "description": "d"}, {"name": "Introduction"}, {"name": 5}]})])
    assert extract_from_text(llm, "text") == [("Gradient Descent", "d")]
    assert extract_from_text(Fake(["x", "y"]), "text") is None       # two failures -> None (counted as failed batch)


def test_parse_edges_validates_names_and_clamps():
    ids = {"loss function": 1, "gradient descent": 2}
    out = parse_edges(json.dumps({"edges": [
        {"concept": "Gradient Descent", "prerequisite": "Loss Functions", "confidence": 7},
        {"concept": "Gradient Descent", "prerequisite": "Made Up"},
        {"concept": "Loss Function", "prerequisite": "loss function"}]}), ids)
    assert out == [(2, 1, 1.0)]


def test_order_adjust_and_cycles():
    assert order_adjust(1.0, (1, 5), (1, 2)) == 1.0          # prerequisite earlier: fine
    assert order_adjust(1.0, (1, 2), (1, 5)) == 0.6          # prerequisite appears later: less trust
    assert order_adjust(1.0, (1, 2), (2, 5)) == 1.0          # different documents: no information
    req = {1: {2}, 2: {3}}                                   # 1 needs 2, 2 needs 3
    assert would_cycle(req, 3, 1) and not would_cycle(req, 1, 3)
