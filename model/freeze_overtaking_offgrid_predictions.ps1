$ErrorActionPreference = "Stop"
python validate_overtaking_offgrid_manifest.py
python predict_overtaking_v5_offgrid.py `
  --model results/overtaking_v5_final_model.joblib `
  --manifest data/overtaking_offgrid_manifest.csv `
  --out results/overtaking_v5_offgrid_predictions_LOCKED.csv
