param([string]$Premise = "pairs", [double]$Threshold = 0.5,
      [string]$Policy = "abstain_if_contradicted", [string]$Prefix = "v2_")
foreach ($c in "A","C","B","D","E","F") {
    python experiments\run_experiment.py --condition $c --n 40 --name "$Prefix$c" `
        --premise-mode $Premise --support-threshold $Threshold --final-policy $Policy
}
