# Dogs vs Cats (DINOv2) - Kaggle Notebook Pipeline

This folder contains a simplified DinoV2 training workflow for Kaggle `dogs-vs-cats-redux-kernels-edition`.

## Scope
- Binary target: `dog=1`, `cat=0`
- Backbone default: `facebook/dinov2-base` (ViT-B/14)
- Validation protocol: single stratified holdout (80/20)
- Runtime targets:
  - single GPU notebook
  - 2xT4 `DataParallel` notebook

## Folder Structure
- `configs/`
  - `paths_kaggle.json`, `paths_local.json` (both support direct folders and zip fallback)
  - `experiments/dinov2_vitb14_single_small.json`
  - `experiments/dinov2_vitb14_single_mid.json`
  - `experiments/dinov2_vitb14_parallel_t4x2_small.json`
  - `experiments/dinov2_vitb14_parallel_t4x2_mid.json`
- `src/`
  - `dinov2_pipeline.py`: shared preprocess/sanity/train pipeline
  - `preprocess_competition_data.py`: zip extraction + manifest creation
  - `sanity_check_random_init.py`: random-init baseline + smoke validation
- `notebooks/`
  - `02_dinov2_training_single_gpu.ipynb`
  - `02_dinov2_training_parallel_t4x2.ipynb`
- `evaluation/`
  - `metrics/`, `plots/`, `predictions/`, `reports/`

## Pipeline Entry Points
1. Preprocess:
```bash
python dogs_vs_cats/src/preprocess_competition_data.py \
  --paths-config dogs_vs_cats/configs/paths_kaggle.json \
  --experiment-config dogs_vs_cats/configs/experiments/dinov2_vitb14_single_small.json
```

Preprocess source priority:
1. `train_dir` + `eval_dir` with class folders (`cats/`,`dogs/`) for pre-split train/eval datasets
2. `train_dir`/`test_dir` if they exist (no unzip needed)
3. otherwise `train.zip`/`test.zip` extraction

2. Sanity:
```bash
python dogs_vs_cats/src/sanity_check_random_init.py \
  --paths-config dogs_vs_cats/configs/paths_kaggle.json \
  --experiment-config dogs_vs_cats/configs/experiments/dinov2_vitb14_single_small.json
```

3. Train:
```bash
python dogs_vs_cats/src/dinov2_pipeline.py \
  --mode train \
  --paths-config dogs_vs_cats/configs/paths_kaggle.json \
  --experiment-config dogs_vs_cats/configs/experiments/dinov2_vitb14_single_small.json
```

## Output Artifacts
For each experiment, outputs are saved under paths from `paths_kaggle.json`:
- metrics CSV (train history, eval history, final val metrics)
- plots (train loss, grad norm, val metrics, confusion matrix)
- predictions CSVs (val + test)
- submission CSV (`id,label` probability of dog)
- JSON reports (preprocess summary, sanity summary, training summary)

## Notebook Format
Use the fixed notebook structure described in:
- `docs/kaggle_format.md`
