# Runs conditions sequentially on the 40 test questions. Safe to re-run: finished questions are skipped.
$conds = "A","C","B","D","E","F"
foreach ($c in $conds) {
    python experiments\run_experiment.py --condition $c --n 40 --name "cond_$c"
}
