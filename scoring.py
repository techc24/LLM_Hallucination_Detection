from typing import Dict, List, Tuple

from .calibration import calibrate_score


def aggregate_hallucination(claim_labels: List[str], evidence_scores: List[float]) -> Tuple[float, str, Dict[str, float]]:
    """
    Aggregate per-claim labels and evidence scores into a single hallucination score.

    Length-normalized weighted aggregation to reduce long-answer bias.
    """
    if not claim_labels:
        empty = {"raw_score": 0.0, "calibrated_score": 0.0, "contradicted_ratio": 0.0, "unknown_ratio": 0.0, "weak_evidence_ratio": 0.0}
        return 0.0, "No claims detected; hallucination score is 0.0 by default.", empty

    contradicted = 0
    unknown = 0
    weak_evidence = 0

    for label, ev in zip(claim_labels, evidence_scores):
        if label == "contradicted":
            contradicted += 1
        elif label == "unknown":
            unknown += 1
        if ev < 0.4:
            weak_evidence += 1

    n_claims = max(len(claim_labels), 1)
    contradicted_ratio = contradicted / n_claims
    unknown_ratio = unknown / n_claims
    weak_evidence_ratio = weak_evidence / n_claims

    # Weighted average over ratios (normalized by claim count).
    raw_score = (
        0.6 * contradicted_ratio +
        0.25 * unknown_ratio +
        0.15 * weak_evidence_ratio
    )
    raw_score = max(0.0, min(1.0, raw_score))
    score = calibrate_score(raw_score)

    explanation_parts = []
    if contradicted:
        explanation_parts.append(f"{contradicted} claim(s) contradicted by evidence")
    if unknown:
        explanation_parts.append(f"{unknown} claim(s) with no clear supporting evidence")
    if weak_evidence:
        explanation_parts.append(f"{weak_evidence} claim(s) with weak evidence (<0.4)")

    if not explanation_parts:
        explanation = "All claims appear supported with reasonable evidence."
    else:
        explanation = "; ".join(explanation_parts) + "."

    explanation = f"Hallucination score: {score:.2f} (raw: {raw_score:.2f}). " + explanation
    breakdown = {
        "raw_score": round(raw_score, 4),
        "calibrated_score": round(score, 4),
        "contradicted_ratio": round(contradicted_ratio, 4),
        "unknown_ratio": round(unknown_ratio, 4),
        "weak_evidence_ratio": round(weak_evidence_ratio, 4),
    }
    return score, explanation, breakdown

