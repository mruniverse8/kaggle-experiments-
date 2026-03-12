# Configs

This folder stores notebook-consumable JSON configs.

## Path Configs
- `paths_kaggle.json`: Kaggle runtime paths (direct dirs, competition zips, extracted dirs, outputs)
- `paths_local.json`: local-machine path template with same schema

`train_dir` and `test_dir` are optional direct-folder inputs.
If those paths exist, preprocess skips zip extraction.
`eval_dir` is an optional labeled evaluation folder (for `training_set`/`test_set` style datasets).
If `train_dir` + `eval_dir` both contain `cats/` and `dogs/`, preprocess uses them directly
without creating a random holdout split.

## Experiment Configs
Primary configs for this simplified pipeline:
- `experiments/dinov2_vitb14_single_small.json`
- `experiments/dinov2_vitb14_single_mid.json`
- `experiments/dinov2_vitb14_parallel_t4x2_small.json`
- `experiments/dinov2_vitb14_parallel_t4x2_mid.json`

## Required Runtime Keys
Each experiment config includes:
- `use_data_parallel`
- `parallel_gpu_ids`
- `amp_dtype`
- `gradient_checkpointing`
- `training.grad_accum_steps`
- `training.log_every_steps`
- `training.eval_every_steps`
- `weights_source`
- `local_weights_path`

## Usage Pattern
1. Load one `paths_*.json`.
2. Load one experiment config from `experiments/`.
3. Run preprocess, sanity, then train via pipeline scripts.
4. Read output summaries from `evaluation/reports/`.
