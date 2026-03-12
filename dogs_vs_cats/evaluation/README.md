# Evaluation Workspace (Lightweight)

This project intentionally keeps evaluation outputs minimal for Kaggle iteration speed.

## Subfolders
- `metrics/`
  - final val metrics CSV
  - train history CSV
  - eval history CSV
- `plots/`
  - train loss curve
  - grad norm curve
  - val AUC/logloss curve
  - final val confusion matrix
- `predictions/`
  - val predictions CSV
  - test predictions CSV
  - Kaggle submission CSV
- `reports/`
  - preprocess summary JSON
  - random-init sanity summary JSON
  - training summary JSON

## Naming Convention
All files are prefixed with `experiment_name`, for example:
- `dinov2_vitb14_single_small_metrics.csv`
- `dinov2_vitb14_single_small_val_predictions.csv`
- `dinov2_vitb14_single_small_submission.csv`
- `dinov2_vitb14_single_small_training_summary.json`
