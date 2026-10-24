"""One-off patch: add final_policy 'abstain_if_contradicted'."""
from pathlib import Path

wf = Path("src/graph/workflow.py")
t = wf.read_text(encoding="utf-8")
old = '''    def n_critic(self, state):
        if state["round"] < self.cfg.max_rounds:
            action = "RE_RETRIEVE"
        elif self.cfg.final_policy == "abstain":
            action = "ABSTAIN"
        else:
            action = "ACCEPT_FLAGGED"
'''
new = '''    def n_critic(self, state):
        contradicted = any(c["verdict"] == "CONTRADICTED" for c in state["claim_results"])
        if state["round"] < self.cfg.max_rounds:
            action = "RE_RETRIEVE"
        elif self.cfg.final_policy == "abstain":
            action = "ABSTAIN"
        elif self.cfg.final_policy == "abstain_if_contradicted" and contradicted:
            action = "ABSTAIN"
        else:
            action = "ACCEPT_FLAGGED"
'''
assert old in t, "n_critic block not found - file differs from expected"
wf.write_text(t.replace(old, new), encoding="utf-8")

ex = Path("experiments/run_experiment.py")
t = ex.read_text(encoding="utf-8")
old2 = 'choices=["abstain", "accept_flagged"]'
assert old2 in t, "final-policy argument not found"
ex.write_text(t.replace(old2, 'choices=["abstain", "accept_flagged", "abstain_if_contradicted"]'), encoding="utf-8")
print("patched OK")
