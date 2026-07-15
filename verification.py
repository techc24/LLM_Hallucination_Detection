import json
import os
from functools import lru_cache
from typing import Dict, List, Tuple
from difflib import SequenceMatcher


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def load_knowledge_bases() -> Dict[str, List[Dict]]:
    return {
        "finance": _load_json(os.path.join(DATA_DIR, "finance.json")),
        "law": _load_json(os.path.join(DATA_DIR, "law.json")),
        "healthcare": _load_json(os.path.join(DATA_DIR, "healthcare.json")),
    }


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.replace(",", " ").replace(".", " ").split() if t.strip()]


def _semantic_similarity(a: str, b: str) -> float:
    """
    Lightweight semantic proxy using normalized sequence similarity.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def retrieve_evidence(domain: str, claim: str, top_k: int = 3) -> List[Dict]:
    kb = load_knowledge_bases().get(domain, [])
    claim_tokens = set(_tokenize(claim))
    scored: List[Tuple[float, Dict]] = []

    for entry in kb:
        text = f"{entry.get('title', '')} {entry.get('description', '')}"
        entry_tokens = set(_tokenize(text))
        if not entry_tokens:
            continue
        overlap = len(claim_tokens & entry_tokens) / len(entry_tokens)
        semantic = _semantic_similarity(claim, text)
        combined = 0.6 * overlap + 0.4 * semantic
        if combined > 0.1:
            scored.append((combined, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:top_k]]


def _detect_contradiction(claim: str, evidence: str) -> bool:
    """
    Extremely simple, domain-agnostic contradiction detector.
    Looks for obvious negation patterns on toy data.
    """
    claim_l = claim.lower()
    evidence_l = evidence.lower()
    neg_words = ["not", "never", "no", "cannot", "can't"]

    claim_has_neg = any(w in claim_l for w in neg_words)
    evidence_has_neg = any(w in evidence_l for w in neg_words)

    if claim_has_neg and not evidence_has_neg:
        return True
    if evidence_has_neg and not claim_has_neg:
        return True
    return False


def infer_nli_relation(claim: str, evidence: str) -> str:
    """
    NLI-style heuristic classifier: entailment / contradiction / neutral.
    """
    if _detect_contradiction(claim, evidence):
        return "contradiction"

    claim_tokens = set(_tokenize(claim))
    evidence_tokens = set(_tokenize(evidence))
    if not claim_tokens:
        return "neutral"

    overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    if overlap >= 0.45:
        return "entailment"
    if overlap <= 0.12:
        return "neutral"
    return "neutral"


def score_claim(domain: str, claim: str) -> Dict:
    """
    Compute label and evidence score for a single claim.
    """
    evidence_entries = retrieve_evidence(domain, claim)
    evidence_texts = [e.get("description", "") for e in evidence_entries]

    if not evidence_entries:
        return {
            "claim": claim,
            "evidence_snippets": [],
            "label": "unknown",
            "evidence_score": 0.0,
            "nli_relation": "neutral",
        }

    best_overlap = 0.0
    for e in evidence_entries:
        text = e.get("description", "")
        claim_tokens = set(_tokenize(claim))
        text_tokens = set(_tokenize(text))
        if not text_tokens:
            continue
        overlap = len(claim_tokens & text_tokens) / len(claim_tokens or {1})
        if overlap > best_overlap:
            best_overlap = overlap

    evidence_score = min(1.0, max(0.0, best_overlap))

    label = "supported"
    top_relation = "neutral"
    for snippet in evidence_texts:
        relation = infer_nli_relation(claim, snippet)
        if top_relation == "neutral":
            top_relation = relation
        if relation == "contradiction":
            label = "contradicted"
            evidence_score = min(evidence_score, 0.3)
            top_relation = "contradiction"
            break
        if relation == "entailment":
            top_relation = "entailment"

    if label != "contradicted" and evidence_score < 0.18:
        label = "unknown"

    return {
        "claim": claim,
        "evidence_snippets": evidence_texts,
        "label": label,
        "evidence_score": evidence_score,
        "nli_relation": top_relation,
    }


def detect_confidence_flags(claim: str, evidence_score: float) -> List[str]:
    """
    Very simple heuristic for confidence–evidence mismatch.
    """
    flags: List[str] = []
    strong_words = ["always", "guaranteed", "definitely", "certainly", "never fails"]
    hedged_words = ["may", "might", "could", "sometimes", "often"]

    text = claim.lower()
    strong = any(w in text for w in strong_words)
    hedged = any(w in text for w in hedged_words)

    if strong and evidence_score < 0.4:
        flags.append("High-confidence wording with weak evidence")
    if hedged and evidence_score > 0.8:
        flags.append("Hedged wording but strong supporting evidence")

    return flags