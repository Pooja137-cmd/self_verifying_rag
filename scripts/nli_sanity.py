"""Check that the NLI model loads, labels are mapped correctly, and measure CPU speed."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.nli import DEFAULT_NLI_MODEL, NLIScorer  # noqa: E402

P = "The Eiffel Tower was completed in 1889 for the World's Fair in Paris."
CASES = [("The Eiffel Tower was completed in 1889.", "entailment"),
         ("The Eiffel Tower was completed in 1925.", "contradiction"),
         ("The tower receives seven million visitors a year.", "neutral")]

nli = NLIScorer()
print("model:", DEFAULT_NLI_MODEL)
t0 = time.perf_counter()
out = nli.score([P] * len(CASES), [c for c, _ in CASES])
print(f"first call (includes load): {time.perf_counter() - t0:.1f}s")
t0 = time.perf_counter()
out = nli.score([P] * 12, [CASES[i % 3][0] for i in range(12)])
print(f"12 pairs: {time.perf_counter() - t0:.2f}s  ({(time.perf_counter() - t0) / 12:.3f}s/pair)")
out = nli.score([P] * len(CASES), [c for c, _ in CASES])
ok = 0
for (c, gold), o in zip(CASES, out):
    pred = max(o, key=o.get)
    ok += pred == gold
    print(f"  gold={gold:<14} pred={pred:<14} {o}")
print(f"\n{ok}/{len(CASES)} sanity cases correct (3 trivial cases; NOT an accuracy estimate)")
