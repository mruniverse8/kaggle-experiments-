# plan_dinov_2: Dogs vs Cats with DINOv2

## 1. Objective and recommendation
Goal: train a high-quality binary classifier (`dog=1`, `cat=0`) using a pretrained DINOv2 backbone.

Primary recommendation:
- Start with **DINOv2 Base (ViT-B/14, ~86M params)**.
- Move to **DINOv2 Large (ViT-L/14)** only after ViT-B baseline is stable and you confirm enough GPU memory/time.

Why:
- Dataset is medium-size (25k labeled images).
- ViT-L often gives marginal gains here unless fine-tuned carefully; compute cost rises sharply.

## 2. Data at the beginning (raw format)
Local dataset layout:
- `dogs-vs-cats/train/` -> 25,000 labeled files (`cat.<id>.jpg`, `dog.<id>.jpg`)
- `dogs-vs-cats/test1/` -> 12,500 unlabeled files (`<id>.jpg`)

Observed image size range is highly variable, so resizing/cropping is mandatory.

### 2.1 Create training manifests
Build two CSV/Parquet manifests:
- `train_manifest.csv`: `filepath,label,id`
- `test_manifest.csv`: `filepath,id`

Label parsing rule:
- filename starts with `cat.` -> label `0`
- filename starts with `dog.` -> label `1`

### 2.2 Split strategy
Use **stratified split** on `train_manifest`:
- `train`: 80%
- `val`: 20%

Optional stronger estimate:
- 5-fold stratified CV (recommended if target is best Kaggle score).

Important:
- Keep `test1/` strictly for final inference/submission.

### 2.3 Data quality checks
Before training:
- detect unreadable/corrupt images
- remove exact duplicates if any
- verify class balance after split

## 3. Input pipeline for DINOv2

## 3.1 Resolution
Use:
- Phase A: `224x224`
- Phase B (optional final polish): `280` or `336` fine-tune for 1-3 short epochs

Reason:
- DINOv2 patch size is 14; 224 is standard and efficient.

### 3.2 Normalization
Use ImageNet mean/std unless your loaded DINOv2 implementation specifies custom transforms:
- mean `[0.485, 0.456, 0.406]`
- std  `[0.229, 0.224, 0.225]`

### 3.3 Augmentations (train only)
Balanced, not extreme:
- RandomResizedCrop(224, scale ~[0.6, 1.0])
- HorizontalFlip(p=0.5)
- ColorJitter (light)
- RandomErasing(p=0.1-0.2)

Validation/test:
- Resize + CenterCrop
- no stochastic augmentation

## 4. Model architecture choice

## 4.1 Backbone
Option A (recommended first):
- `dinov2_vitb14` (embedding dim ~768)

Option B (second step):
- `dinov2_vitl14` (embedding dim ~1024)

## 4.2 Classification head
For binary classification, best starting head is simple:
- `LayerNorm(embed_dim)`
- `Linear(embed_dim -> 1)`

Output:
- one logit
- probability with sigmoid during inference

Loss:
- `BCEWithLogitsLoss`

### 4.3 Should we project into another space?
Short answer:
- **Usually not necessary** for this task.

Details:
- A heavy projection MLP is often unnecessary for binary cats-vs-dogs and can overfit.
- If needed, use a small projection only:
  - `LayerNorm -> Linear(embed_dim, 256) -> GELU -> Dropout(0.2) -> Linear(256,1)`
- Use this only if validation metrics plateau with the simple head and you need extra capacity.

Conclusion:
- Start with **no projection**.
- Add small projection MLP only as an experiment, not as default.

## 5. Training strategy (step by step)

### Step 1: Linear probe (fast sanity check)
- freeze full backbone
- train only classification head
- epochs: 3-5
- lr (head): `1e-3`

Expected output:
- quick baseline, validates data pipeline and labels.

### Step 2: Partial fine-tuning
- unfreeze last transformer blocks
  - ViT-B: last 4 blocks
  - ViT-L: last 6 blocks
- discriminative LR:
  - backbone: `1e-5` to `3e-5`
  - head: `3e-4` to `1e-3`
- epochs: 8-15

### Step 3: Full fine-tuning (if stable)
- unfreeze all blocks
- very low backbone LR + layer-wise LR decay (LLRD)
  - base lr: `1e-5` backbone
  - head lr: `1e-4`
  - layer decay: ~0.75-0.85
- epochs: 5-10 (early stopping)

## 5.1 Optimizer/scheduler and stability
- optimizer: `AdamW`
- weight decay: `0.05`
- scheduler: cosine decay + warmup (5-10% steps)
- AMP: fp16/bf16
- gradient clip: `1.0`
- batch size:
  - ViT-B: start with 32 (reduce if needed)
  - ViT-L: start with 8-16 + gradient accumulation

## 6. Metrics and model selection
Primary metrics:
- ROC-AUC
- Log loss (BCE)

Secondary:
- accuracy, F1

Select checkpoint by:
- best validation ROC-AUC (or lowest val log loss)

## 7. Inference and submission
For `test1/`:
- output `id, label`
- where `label = P(dog)` from sigmoid(logit)

Optional test-time augmentation (TTA):
- average predictions from original + horizontal flip
- add multi-crop TTA only if it improves validation reliably

## 8. Compute/cost notes
ViT-B:
- practical default, strong tradeoff quality/cost

ViT-L:
- much heavier memory/compute
- only worth it if:
  - GPU budget is sufficient
  - ViT-B has converged and you need final incremental gain

## 9. Experiment matrix (concrete)
Run these in order:
1. `B14 + linear head + frozen backbone`
2. `B14 + linear head + partial unfreeze`
3. `B14 + linear head + full fine-tune + LLRD`
4. `B14 + small projection MLP head` (only if #3 plateaus)
5. `L14 + linear head + partial unfreeze` (only if resources allow)

## 10. What is not a good idea here
- Training DINOv2 from scratch on 25k images
- Starting directly with ViT-L full fine-tune before validating pipeline
- Using overly aggressive augmentations early (can hurt fine-grained features)
- Forcing a large projection head without evidence it improves validation

## 11. Final decision rule
If you need one model today:
- choose **DINOv2 ViT-B/14 + simple LayerNorm+Linear head + staged unfreezing**.

If you have extra compute and need last-mile gain:
- test **ViT-L/14** after ViT-B baseline is fully tuned.
