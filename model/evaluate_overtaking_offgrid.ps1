param(
  [Parameter(Mandatory=$true)][string]$CasesDir
)
$ErrorActionPreference = "Stop"
python evaluate_overtaking_v5_offgrid.py `
  --predictions results/overtaking_v5_offgrid_predictions_LOCKED.csv `
  --cases-dir $CasesDir `
  --out-prefix results/overtaking_v5_offgrid_independent
