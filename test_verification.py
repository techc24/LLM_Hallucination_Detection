from app.verification import detect_confidence_flags, infer_nli_relation, score_claim


def test_score_claim_unknown_when_no_overlap():
    result = score_claim("finance", "The moon is made of cheese")
    assert result["label"] in {"unknown", "contradicted"}


def test_detect_confidence_flags_high_conf_low_evidence():
    flags = detect_confidence_flags("This is always true in every case", evidence_score=0.1)
    assert any("High-confidence wording" in f for f in flags)


def test_nli_relation_contradiction():
    relation = infer_nli_relation(
        "Antibiotics are effective against viral infections.",
        "Antibiotics are not effective against viral infections such as the common cold or influenza.",
    )
    assert relation == "contradiction"

