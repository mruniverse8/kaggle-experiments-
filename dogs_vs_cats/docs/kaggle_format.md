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
- if manifests already exist, skip preprocess and continue to sanity/training
- preprocess mode now prefers direct dataset folders (`train_dir` / `test_dir`) when available
- if `train_dir` and `eval_dir` are class-folder datasets (`cats`,`dogs`), it uses them as fixed train/eval manifests
- if direct folders are missing, it falls back to extracting `train.zip` / `test.zip`

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

## 8. Export Artifacts Cell (Producer Notebooks)
- run:

```bash
python dogs_vs_cats/src/export_experiment_artifacts.py \
  --paths-config <paths_cfg> \
  --experiment-config <cfg_path>
```

- write deterministic reusable bundle under `artifacts_output_dir`:
  - `bundle_index.json`
  - `<experiment_name>/manifest.json`
  - copied checkpoints/evaluation files/configs

## 03 Hypothesis Notebook Format (Consumer)
- bootstrap repo + dependencies
- load `PATHS_CFG` and `ARTIFACT_INPUT_ROOT`
- read uploaded `bundle_index.json`
- select experiments (env-controlled)
- validate required result files
- compare metrics directly from uploaded manifests/results (no retrain)
- optional fallback:
  - `ENABLE_FALLBACK_EVAL=1`
  - calls evaluate-only mode:

```bash
python dogs_vs_cats/src/dinov2_pipeline.py \
  --mode evaluate \
  --paths-config <paths_cfg> \
  --experiment-config <uploaded_exp_cfg> \
  --checkpoint-path <uploaded_best_ckpt> \
  --summary-suffix eval_only
```

## Notes
- Keep evaluation lightweight: no OOF/fold reports by default.
- Default validation protocol is single stratified holdout (80/20).
- For 2xT4 notebooks use `DataParallel` with conservative batch size + grad accumulation.
