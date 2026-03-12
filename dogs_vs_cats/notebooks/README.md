# Notebook Plan (Simplified)

Notebook-first run order for DinoV2 Dogs vs Cats:

1. `02_dinov2_training_single_gpu.ipynb`
- bootstrap repo + install requirements
- load config (`single_small`/`single_mid`)
- preprocess competition zip files into manifests
- run random-init sanity check
- train DinoV2 ViT-B/14 on single GPU
- save metrics/plots/predictions/submission

2. `02_dinov2_training_parallel_t4x2.ipynb`
- same flow as above
- config defaults to `parallel_t4x2_small`
- uses `DataParallel` over GPUs `[0,1]`
- memory-safe defaults (`fp16`, grad accumulation, checkpointing)

## Config Switching
Use environment variable `CFG_PATH` to swap profiles:
- `dogs_vs_cats/configs/experiments/dinov2_vitb14_single_small.json`
- `dogs_vs_cats/configs/experiments/dinov2_vitb14_single_mid.json`
- `dogs_vs_cats/configs/experiments/dinov2_vitb14_parallel_t4x2_small.json`
- `dogs_vs_cats/configs/experiments/dinov2_vitb14_parallel_t4x2_mid.json`

## Required Paths
These notebooks expect Kaggle competition files:
- `/kaggle/input/competitions/dogs-vs-cats-redux-kernels-edition/train.zip`
- `/kaggle/input/competitions/dogs-vs-cats-redux-kernels-edition/test.zip`
- `/kaggle/input/competitions/dogs-vs-cats-redux-kernels-edition/sample_submission.csv`
