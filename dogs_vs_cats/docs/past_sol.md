# Past Solution Analysis: `classifier-dogs-or-cats.ipynb`

## 1. What this notebook is solving
The notebook implements a baseline image classifier for a **2-class problem** (`cats` vs `dogs`) using a **custom CNN from scratch** in TensorFlow/Keras.

It covers:
- data loading + augmentation
- quick EDA/visual diagnostics
- CNN architecture design
- feature visualization (feature maps, filter visualization, Grad-CAM)
- training and evaluation
- single-image inference

## 2. Dataset and data flow (step by step)

### 2.1 Official dataset description (`dogs-vs-cats` folder)
From the local dataset and your description:
- `train/` contains **25,000 labeled images**
  - label is encoded in filename: `cat.<id>.jpg` or `dog.<id>.jpg`
  - verified counts: `12,500` cats + `12,500` dogs
- `test1/` contains **12,500 unlabeled images**
  - filenames are numeric IDs (`<id>.jpg`)
  - target output for competition format is `P(dog)` (probability of class `dog`)

### 2.2 Real image width/height behavior (verified from local files)
Original images are **not fixed-size** and vary significantly.

Measured from local files:
- `train/`: `25,000` readable images, size range **42x32** up to **1050x768**
- `test1/`: `12,500` readable images, size range **37x44** up to **500x500**

Most common train resolutions:
- `500x374` (2955 images)
- `499x375` (2912 images)
- `375x499` (261 images)
- `499x333` (229 images)
- `374x500` (227 images)

This confirms the model needs a resize step before batching.

### 2.3 Paths and folder structure used inside the notebook
Configured paths:
- `TRAIN_PATH = '../input/dogs-cats-images/dog vs cat/dataset/training_set'`
- `TEST_PATH  = '../input/dogs-cats-images/dog vs cat/dataset/test_set'`

The code assumes `flow_from_directory` layout:
- `training_set/cats`, `training_set/dogs`
- `test_set/cats`, `test_set/dogs`

So the notebook is using a **pre-split variant** (class folders), not the raw Kaggle `train/test1` layout directly.

### 2.4 Classes and sample counts (from notebook output)
- `Found 8000 images belonging to 2 classes.` (train)
- `Found 2000 images belonging to 2 classes.` (test)
- Classes detected: `['cats', 'dogs']`

From evaluation support (classification report):
- test support = `1000` cats + `1000` dogs (balanced test set)

### 2.5 Image W/H and tensor shape handling
- Configured size: `IMAGE_SIZE = 128`
- Every image is resized to **128 x 128**
- Color channels: **3 (RGB)**
- Effective model input: **(128, 128, 3)**

With `BATCH_SIZE = 32`, generator batches are typically:
- train batch: `(32, 128, 128, 3)` images + one-hot labels `(32, 2)`
- test batch: same format

### 2.6 Pixel scaling and augmentation
Train generator (`ImageDataGenerator`) applies:
- `rescale=1./255`
- `rotation_range=20`
- `width_shift_range=0.15`
- `height_shift_range=0.15`
- `shear_range=0.1`
- `zoom_range=0.15`
- `horizontal_flip=True`
- `fill_mode='nearest'`

Test generator applies only:
- `rescale=1./255`

### 2.7 Data diagnostics done in notebook
- class distribution plot
- 4x4 sample image grid
- augmentation visualization from one image
- per-class RGB intensity histogram sampling

## 3. Current small approach (solution design)

### 3.1 Model architecture
Model: `DogsCats_CNN` (custom CNN, 4 conv blocks + classification head)

Blocks:
1. Conv(32) -> BN -> Conv(32) -> BN -> MaxPool -> Dropout(0.25)
2. Conv(64) -> BN -> Conv(64) -> BN -> MaxPool -> Dropout(0.25)
3. Conv(128) -> BN -> Conv(128) -> BN -> MaxPool -> Dropout(0.3)
4. Conv(256) -> BN -> MaxPool -> Dropout(0.3)

Head:
- GlobalAveragePooling2D
- Dense(256, relu) -> BN -> Dropout(0.5)
- Dense(2, softmax)

Params (from summary):
- Total: **652,322**
- Trainable: **650,402**
- Non-trainable: **1,920**

### 3.2 Feature extraction / interpretability utilities
Implemented:
- intermediate activation extraction for `conv1_1`, `conv2_1`, `conv3_1`, `conv4_1`
- first-layer kernel visualization (`conv1_1` 3x3 filters)
- Grad-CAM overlay from final conv block (`conv4_1`)

Reported activation tensor shapes:
- `conv1_1 -> (1, 128, 128, 32)`
- `conv2_1 -> (1, 64, 64, 64)`
- `conv3_1 -> (1, 32, 32, 128)`
- `conv4_1 -> (1, 16, 16, 256)`

### 3.3 Training setup
Compile:
- optimizer: `Adam`
- loss: `categorical_crossentropy`
- metric: `accuracy`
- LR schedule: `ExponentialDecay(1e-3, decay_steps=500, decay_rate=0.9)`

Callbacks:
- `EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)`
- `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)`
- `ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')`

Training call specifics:
- `epochs = 20`
- `steps_per_epoch = 8000 // 32 = 250`
- `validation_steps = 2000 // 32 = 62`

## 4. Observed performance in this notebook run

### 4.1 Learning trajectory
From logged epochs:
- train accuracy improved from **0.5477** to **0.8894**
- val accuracy improved from **0.5030** to around **0.8901**
- best observed val accuracy in logs: **0.8952** (epoch 17)

### 4.2 Final reported evaluation
- Test Loss: **0.2709**
- Test Accuracy: **88.90%**

Classification report:
- cats: precision **0.89**, recall **0.89**, f1 **0.89**
- dogs: precision **0.89**, recall **0.88**, f1 **0.89**
- overall accuracy: **0.89** on 2000 samples

Misclassification count:
- **222 / 2000 (11.1%)**

Single custom-image inference example shown:
- predicted class: `cats`
- confidence: **99.0%**

## 5. Technical assessment of the current approach

### 5.1 What is good
- clean baseline CNN with regularization (BN + Dropout)
- reasonable augmentation policy for small/medium image dataset
- strong notebook pedagogy: EDA + interpretability + error analysis
- achieved near-89% test accuracy with a lightweight custom model

### 5.2 Methodological issues / risks
1. **Test leakage in training loop**
- `test_generator` is used as `validation_data` during training, then used again for final test metrics.
- This weakens the purity of final generalization estimates.

2. **Validation coverage per epoch is truncated**
- `validation_steps = 2000 // 32 = 62` means only `62 * 32 = 1984` test images used per validation epoch.
- 16 images are skipped in per-epoch validation metrics.

3. **No dedicated validation split from training set**
- Better practice: split training data into train/val and keep test set untouched until final evaluation.

4. **Binary problem treated as 2-class categorical setup**
- Current setup is valid, but a binary head (`1` sigmoid + `binary_crossentropy`) could simplify the pipeline.

5. **Raw Kaggle dataset is not consumed directly**
- Local raw data is `train/` + `test1/` with filename labels, but notebook expects `training_set/` + `test_set/` class directories.
- Reproducing this notebook from raw files requires a preprocessing split/organize step.

## 6. Bottom line
This notebook is a **good baseline solution** for a small dogs-vs-cats classifier:
- standardized 128x128 RGB inputs
- moderate augmentation
- custom 652k-parameter CNN
- final performance around **88.9% test accuracy** on its 8k/2k pre-split data setup

The main improvement needed for a more reliable experiment is to fix data protocol:
- create a true validation split from training data
- keep test set strictly for one-time final evaluation.
