# DZ3: Dogs vs Cats Baseline

## Files in this folder
- `classifier-dogs-or-cats.ipynb`: baseline CNN notebook (EDA + training + evaluation + inference)
- `past_sol.md`: technical audit of the current solution and dataset details
- `plan_dinov_2.md`: technical migration plan to DINOv2 (ViT-B/14 first, ViT-L/14 optional)
- `dogs-vs-cats/`: local Kaggle dataset files (`train/`, `test1/`)

## Dataset format used locally
The local dataset in `dogs-vs-cats/` follows the original Kaggle structure:
- `train/` -> 25,000 labeled images (`cat.<id>.jpg`, `dog.<id>.jpg`)
- `test1/` -> 12,500 unlabeled images (`<id>.jpg`)

The notebook itself is configured for a pre-split `training_set/` and `test_set/` directory layout.

## Current baseline result (from notebook run)
- Input size: `128x128x3`
- Model: custom CNN, `652,322` parameters
- Reported test accuracy: `88.90%`
- Misclassified: `222 / 2000` (`11.1%`)

## Notes
- The full step-by-step findings are documented in `past_sol.md`.
- Main methodology issue to fix next: avoid using test data as validation during training.
