# Known limitations and bugs (as of the v2 baseline; update as fixed)

Evidence for each item is in the saved per-question logs (evaluation/results/*.jsonl).

1. Sentence splitter bug (v2 results, `cond_*`): claims were split at "Vol. 2" and at middle initials
   ("Robert Z. Leonard" -> "Leonard is younger..."), producing spurious CONTRADICTED verdicts and correct answers
   rejected. Fixed for new runs in `src/retrieval/sentences.py`; old `cond_*` results were NOT regenerated.
2. Whole-chunk NLI premises: near-verbatim copies of evidence received entailment ~0.01 (e.g. the Jihad record),
   so correct answers were rejected. v3 `windows` mode addresses this; effectiveness is tested in
   `scripts/verifier_v3_dev.py` (NOT YET MEASURED until run).
3. Claim threshold: the value used for v2 runs came from 22 dev answers (Youden J) and was degenerate (0.05); claims
   with entailment 0.06-0.11 were accepted.
4. Contradiction was not relevance-gated in v1/v2 (an unrelated chunk could yield max contradiction ~1.0 while
   another premise entailed the claim). v3 gates contradiction on lexical overlap with the claim.
5. Answer/statement inconsistency: the verified claim is the generator's one-sentence statement, which may omit or
   contradict the short answer (Sinclair Hill: statement said Leonard was younger, answer said Hill). v3 logs
   `answer_in_claims` but does not enforce it.
6. Generator abstained on every yes/no comparison question in the first slice of results (3 of 3).
7. `revision_rate` / `re_retrieval_rate`: before v3 no answer revision existed (only re-retrieval plus
   regeneration), and re-retrieval counts mixed evidence-triggered and critic-triggered events.
   `scripts/audit_results.py` separates them.
8. Latency: `latency_s` is a standalone estimate from cached LLM times; `wall_s` is actual time. Quote LLM calls,
   tokens and standalone time, not actual wall time.
9. Test-set contamination: v3 fixes were designed after reading test-question outputs (questions 0-39). Dev work
   used questions 40-99; risk calibration used questions 40-199 (retrieval labels only). Questions 100-139 are the
   confirmation slice and overlap the calibration pool for retrieval only.
10. Risk signal is weak (calibration AUC 0.602; LOW and MEDIUM failure rates equal), so MEDIUM is not distinguished.
11. Faithfulness evaluation (independent NLI model) flagged ~11 of 13 correct answers as unfaithful for condition A;
    it is a noisy proxy, partly circular for verified conditions.
12. Scale: 40 test questions, one 3B local model, one dataset; confidence intervals are wide.
13. Live-web mode and the Streamlit demo: accuracy NOT YET MEASURED; web pages are untrusted input and NLI does not
    defend against prompt injection.
