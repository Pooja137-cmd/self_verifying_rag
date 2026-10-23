from src.evaluation.answer_metrics import contains_gold, exact_match, f1_score, is_abstention, normalize_answer


def test_normalize():
    assert normalize_answer("The Eiffel, Tower!") == "eiffel tower"


def test_em_and_contains():
    assert exact_match("The Paris", "paris") == 1.0
    assert exact_match("in Paris, France", "paris") == 0.0
    assert contains_gold("in Paris, France", "paris") == 1.0


def test_f1():
    assert f1_score("paris france", "paris") == 2 * 0.5 * 1.0 / 1.5
    assert f1_score("london", "paris") == 0.0
    assert f1_score("", "") == 1.0


def test_abstention():
    assert is_abstention("Insufficient information.")
    assert not is_abstention("Paris")