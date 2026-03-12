# Notebook Plan (Simplified)

Notebook-first run order for DinoV2 Dogs vs Cats (producer/consumer):

1. `02_dinov2_training_single_gpu.ipynb`
- bootstrap repo + install requirements
- load config (`single_small`/`single_mid`)
- preprocess competition zip files into manifests
- run random-init sanity check
- train DinoV2 ViT-B/14 on single GPU
- save metrics/plots/predictions/submission
- export reusable artifacts bundle (`manifest.json` + `bundle_index.json`)

2. `02_dinov2_training_parallel_t4x2.ipynb`
- same flow as above
- config defaults to `parallel_t4x2_small`
- uses `DataParallel` over GPUs `[0,1]`
- memory-safe defaults (`fp16`, grad accumulation, checkpointing)
- exports the same reusable bundle format

3. `03_dinov2_augmentation_hypothesis_parallel_t4x2.ipynb`
- consumer notebook: compares uploaded artifacts from `/kaggle/input/...`
- default behavior: no training
- optional fallback evaluate-only mode from uploaded checkpoints (`ENABLE_FALLBACK_EVAL=1`)

4. `04_dinov2_linear_probing_comparison_parallel_t4x2.ipynb`
- producer notebook: trains two runs in sequence on Kaggle GPU
- run 1: `dinov2_vitb14_linear_probe` (backbone frozen, train classifier head only)
- run 2: `dinov2_vitb14_parallel_t4x2_small` (baseline from notebook 02)
- compares both runs with a unified "special" figure and summary table
- exports artifacts for both runs into the bundle format

## Resilience Behavior (Parallel Runs)
- training/evaluation can skip recoverable CUDA/NCCL/DataParallel failing batches
- recovery can automatically disable DataParallel and continue on single GPU
- checkpoints are still updated (`best.pt` / `last.pt`) when possible
- summaries include `resilience` stats and optional `error_events_json` with recovery details

## Config Switching (02 Producer)
Use environment variable `CFG_PATH` to swap profiles:
- `dogs_vs_cats/configs/experiments/dinov2_vitb14_single_small.json`
- `dogs_vs_cats/configs/experiments/dinov2_vitb14_single_mid.json`
- `dogs_vs_cats/configs/experiments/dinov2_vitb14_parallel_t4x2_small.json`
- `dogs_vs_cats/configs/experiments/dinov2_vitb14_parallel_t4x2_mid.json`

## Required Paths (02 Producer)
Training notebooks expect Kaggle competition files:
- `/kaggle/input/competitions/dogs-vs-cats-redux-kernels-edition/train.zip`
- `/kaggle/input/competitions/dogs-vs-cats-redux-kernels-edition/test.zip`
- `/kaggle/input/competitions/dogs-vs-cats-redux-kernels-edition/sample_submission.csv`

## Required Paths (03 Consumer)
- `ARTIFACT_INPUT_ROOT` should point to uploaded bundle dataset root containing `bundle_index.json`.
