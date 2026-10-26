# run_overtaking_v5.ps1
# 在模型目录、已配置 Python 环境后运行。

$ErrorActionPreference = "Stop"

Write-Host "=== 0) 快速检查（环境与文件路径） ===" -ForegroundColor Cyan
python p3_overtaking_v5.py `
  --loads loads_all.npz `
  --extra-desc data/shared_overtaking_rich.h5 `
  --protocol t-blocked `
  --fast `
  --no-fit-final `
  --out-prefix results/overtaking_v5_smoke

Write-Host "=== 1) v5：pair-random + t-blocked + 最终模型训练 ===" -ForegroundColor Cyan
python p3_overtaking_v5.py `
  --loads loads_all.npz `
  --extra-desc data/shared_overtaking_rich.h5 `
  --protocol all `
  --out-prefix results/overtaking_v5

Write-Host "=== 2) 结构化留出 70% 充装率 ===" -ForegroundColor Cyan
python p3_overtaking_v5.py `
  --loads loads_all.npz `
  --extra-desc data/shared_overtaking_rich.h5 `
  --holdout-col fill_level `
  --holdout-value 0.7 `
  --no-fit-final `
  --out-prefix results/overtaking_v5_fill07

Write-Host "=== 完成。输出文件 ===" -ForegroundColor Green
Write-Host "results/overtaking_v5_paircv_summary.csv"
Write-Host "results/overtaking_v5_tblocked_summary.csv"
Write-Host "results/overtaking_v5_final_selection.csv"
Write-Host "results/overtaking_v5_final_model.joblib"
Write-Host "results/overtaking_v5_fill07_summary.csv"
