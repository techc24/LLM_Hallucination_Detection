from app.claims import rule_based_claim_split


def test_rule_based_claim_split_basic():
    text = "Hypertension increases the risk of stroke. It can be managed with lifestyle changes and medication."
    claims = rule_based_claim_split(text)
    assert len(claims) == 2
    assert "Hypertension increases the risk of stroke" in claims[0]

