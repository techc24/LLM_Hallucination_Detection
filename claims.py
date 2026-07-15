from typing import List


def rule_based_claim_split(answer: str) -> List[str]:
    """
    Very simple rule-based splitter used as a fallback and for tests.
    Splits on periods and filters out obviously empty fragments.
    """
    raw_parts = answer.replace("\n", " ").split(".")
    claims = [p.strip() for p in raw_parts if p.strip()]
    return claims

