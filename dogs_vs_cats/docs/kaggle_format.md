# Kaggle Notebook Format (Dogs vs Cats DinoV2)

Use this fixed cell order for all training notebooks in this project.

## 1. Bootstrap Repo Cell
- set `PYTORCH_CUDA_ALLOC_CONF`
- clone repo into `/kaggle/working/kaggle-experiments-` if missing
- `git fetch --all`, `git checkout <branch>`, `git reset --hard origin/<branch>`
- `pip install -U pip`
- `pip install -r requirements.txt`
- print git branch + commit

## 2. Config Load Cell
- set `PATHS_CFG` and `CFG_PATH` (env-overridable)
- read JSON config files
- print selected config paths

## 3. Preprocess Cell
- run:

```bash
python dogs_vs_cats/src/preprocess_competition_data.py \
  --paths-config <paths_cfg> \
  --experiment-config <cfg_path>
```

- verify train/val/test manifest files exist

## 4. Sanity Cell
- run:

```bash
python dogs_vs_cats/src/sanity_check_random_init.py \
  --paths-config <paths_cfg> \
  --experiment-config <cfg_path>
```

- ensure summary reports `"passed": true`

## 5. Training Cell
- call pipeline train mode:

```bash
python dogs_vs_cats/src/dinov2_pipeline.py \
  --mode train \
  --paths-config <paths_cfg> \
  --experiment-config <cfg_path>
```

## 6. Plot/Preview Cell
- display saved training curves:
  - train loss
  - val auc/logloss
  - grad norm
  - final confusion matrix

## 7. Artifact Summary Cell
- print paths for:
  - best checkpoint
  - metrics CSVs
  - val predictions CSV
  - test predictions CSV
  - submission CSV
  - summary JSON report

## Notes
- Keep evaluation lightweight: no OOF/fold reports by default.
- Default validation protocol is single stratified holdout (80/20).
- For 2xT4 notebooks use `DataParallel` with conservative batch size + grad accumulation.
