from app.scoring import aggregate_hallucination


def test_aggregate_hallucination_all_supported():
    labels = ["supported", "supported"]
    scores = [0.9, 0.8]
    value, explanation, breakdown = aggregate_hallucination(labels, scores)
    assert 0.0 <= value <= 0.3
    assert "raw_score" in breakdown
    assert "All claims appear supported" in explanation or "Hallucination score" in explanation


def test_aggregate_hallucination_contradicted_and_unknown():
    labels = ["contradicted", "unknown"]
    scores = [0.2, 0.1]
    value, explanation, breakdown = aggregate_hallucination(labels, scores)
    assert value > 0.2
    assert breakdown["contradicted_ratio"] > 0
    assert "contradicted" in explanation


def test_normalized_scoring_reduces_length_bias():
    short_labels = ["contradicted", "supported"]
    short_scores = [0.1, 0.9]
    long_labels = short_labels + ["supported"] * 8
    long_scores = short_scores + [0.9] * 8
    short_value, _, _ = aggregate_hallucination(short_labels, short_scores)
    long_value, _, _ = aggregate_hallucination(long_labels, long_scores)
    assert long_value <= short_value

