from src.evaluation.audit import answer_changed, correction_outcome, retrieval_triggers


def H(*items):
    return [dict(step=s, **kw) for s, kw in items]


def test_retrieval_triggers_split():
    hist = H(("retrieve", {}), ("assess", {}), ("re_retrieve", {}), ("generate", {"answer": "a"}),
             ("verify_claims", {}), ("critic", {}), ("re_retrieve", {}), ("generate", {"answer": "b"}))
    assert retrieval_triggers(hist) == (1, 1)
    assert retrieval_triggers(H(("retrieve", {}), ("generate", {"answer": "a"}))) == (0, 0)


def rec(first, last, abstained, history_extra, gold="Paris"):
    hist = H(("generate", {"answer": first}), *history_extra)
    return {"gold_answer": gold, "abstained": abstained, "answer": last, "history": hist}


def test_correction_outcomes():
    crit = [("critic", {}), ("revise", {"answer": "Paris"})]
    assert correction_outcome(rec("London", "Paris", False, crit)) == "fixed_wrong"
    crit2 = [("critic", {}), ("revise", {"answer": "London"})]
    assert correction_outcome(rec("Paris", "London", False, crit2)) == "broke_correct"
    assert correction_outcome(rec("Paris", "Paris", False, [("critic", {})])) == "kept_correct"
    assert correction_outcome(rec("London", "Berlin", False, [("critic", {})])) == "still_wrong"
    assert correction_outcome(rec("London", "London", True, [("critic", {})])) == "abstained_after_wrong_first"
    assert correction_outcome(rec("Paris", "Paris", True, [("critic", {})])) == "abstained_after_correct_first"
    assert correction_outcome(rec("Paris", "Paris", False, [])) == "no_correction"


def test_answer_changed():
    assert answer_changed(rec("London", "Paris", False, [("revise", {"answer": "Paris"})]))
    assert not answer_changed(rec("Paris", "Paris", False, [("generate", {"answer": "paris"})]))
