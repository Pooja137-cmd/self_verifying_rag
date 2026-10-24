param([string]$Prefix = "v3_", [int]$Offset = 0, [int]$N = 40, [switch]$Revise)
$thr = "evaluation\results\claim_threshold_v3.json"
foreach ($c in "A","B","C","D","E","F") {
    $a = @("experiments\run_experiment.py","--condition",$c,"--n","$N","--offset","$Offset",
           "--name","$Prefix$c","--premise-mode","windows","--claim-threshold-file",$thr)
    if ($Revise) { $a += "--revise" }
    python @a
}
