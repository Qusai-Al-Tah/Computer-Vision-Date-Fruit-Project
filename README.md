# Deep Learning for Date Fruit Variety Classification and Quality Grading

**Multi-Task Learning and CNN Ensembles on the Date Fruit Dataset for Inspection and Grading**

Omar Abdalla · Qusai Al Tah — Computer Vision Semester Project, American University of Sharjah

This project builds an automated date fruit inspection system that performs two tasks:

1. **Variety classification** — 4 classes: Aseel, Fasli Toto, Gajar, Kupro (all 3,004 original images)
2. **Quality grading** — 3 classes: Grade-1 / Grade-2 / Grade-3 (restricted to Gajar and Kupro, 2,120 images)

### The key methodological finding

Exploratory analysis revealed that **Aseel and Fasli Toto contain *only* Grade-1 images**. A grading model trained on all four varieties could therefore "cheat" by recognizing the variety and predicting Grade-1, without learning any real quality cues. To remove this shortcut, the grading task is restricted to the two varieties (Gajar, Kupro) that contain all three grades. `dataset.py` enforces this automatically for the `grade` and `multi` tasks.

### Headline results (held-out test set)

| Task | Best single model | Ensemble (soft voting) |
|---|---|---|
| Variety (4-class) | **VGG-16: 100.00% acc / 1.000 macro-F1** | 99.56% / 0.9949 |
| Grading (3-class) | GoogLeNet: 75.79% acc / 0.7599 macro-F1 | **77.36% / 0.7787** |

The three grading CNNs disagreed on 102 of 318 test images; the ensemble correctly resolved **67 of the 100** images where at least one model was right and another wrong.

---

## Repository structure

```
.
├── src/
│   ├── dataset.py        # Data indexing, confound filtering, stratified splits, augmentation
│   ├── model.py          # Backbones + single/multi-head architectures (DateNet)
│   ├── train.py          # Training loop: AMP, early stopping, all ablation flags
│   ├── evaluate.py       # Test-set metrics, confusion matrices, saved softmax outputs
│   ├── ensemble.py       # Soft/weighted voting + disagreement analysis
│   ├── plot_curves.py    # Combined training/accuracy curve figures
│   ├── run_all.ps1       # Reproduces all 13 experiments (Windows PowerShell)
│   ├── figures/          # Generated curve/analysis figures (grading)
│   ├── figures_variety/  # Generated curve figures (variety)
│   ├── ensemble_grade.json    # Ensemble results + disagreement analysis (grading)
│   ├── ensemble_variety.json  # Ensemble results + disagreement analysis (variety)
│   └── runs/             # One folder per experiment (see "Output files" below)
├── report/               # LaTeX source of the project report (IEEE template)
├── requirements.txt
└── README.md
```

Trained checkpoints (`best.pt`, ~460 MB total) are **not** committed to the repo; they are attached to the [GitHub Release](../../releases) — download and drop each into its matching `src/runs/<run_name>/` folder to run evaluation without retraining.

---

## What each source file does

### `dataset.py` — data ingestion, cleaning, and augmentation
- `build_index(root)` walks the dataset directory (`<root>/<Variety>/<Size>/<Grade-N>/*.jpg`) and builds a pandas DataFrame of every **original** image with its variety, size, and grade labels. The dataset authors' pre-augmented images are excluded entirely — using them would risk train/test leakage (augmented copies of one fruit landing in different splits).
- `filter_for_task(df, task)` implements the confound fix: for `grade`/`multi` tasks it keeps only varieties with more than one distinct grade (Gajar, Kupro).
- `make_splits(df, seed)` produces a stratified 70/15/15 train/val/test split. Stratification is done on the joint `(variety, size, grade)` cell so even the smallest subgroup (Kupro Grade-3, 116 images) is proportionally represented in all three splits.
- `make_transform(augment, img_size)` defines preprocessing. Train split: `RandomResizedCrop(scale 0.75–1.0)` (scaling), horizontal **and** vertical flips (a date has no canonical "up"), and `RandomRotation(30°)`, followed by ImageNet normalization. Val/test splits: deterministic resize + normalize only.
- `DateFruitDataset` returns `(image, variety_idx, grade_idx)` triples; single-task training simply ignores the label it doesn't need.
- `prepare(...)` ties it all together and also computes **inverse-frequency class weights** for the weighted cross-entropy loss.

### `model.py` — architectures
- `_make_backbone(name)` loads a torchvision backbone with ImageNet weights (`resnet18`, `resnet34`, `resnet50`, `vgg16` (batch-norm variant), `googlenet`) and strips its classifier, returning the feature extractor and feature dimension. GoogLeNet's auxiliary heads are disabled for fine-tuning simplicity.
- `DateNet` attaches one or two heads (Dropout → Linear) on the shared backbone:
  - `task="variety"` or `task="grade"` → a single head (single-task baseline),
  - `task="multi"` → both heads on one shared backbone. This mirrors the classification + localization dual-head pattern from detection architectures, with the second head predicting quality grade instead of box coordinates.
- `freeze_backbone=True` disables gradients on the backbone (used by the frozen-features ablation).
- A `scratch` CNN (trained from random init) serves as the no-transfer-learning baseline.

### `train.py` — training
Every design decision is a CLI flag, so each of the 13 experiments is a single command (see `run_all.ps1`). Pipeline per run:
1. Seeded, stratified data preparation (`--seed 42` everywhere → identical splits across all runs, which is what makes the ensemble valid).
2. Model built per `--task` / `--backbone` / ablation flags (`--no-aug`, `--no-dropout`, `--freeze`, `--grade-loss-weight`).
3. Weighted cross-entropy per head; multi-task loss = `L_variety + λ · L_grade`.
4. AdamW (lr 3e-4, weight decay 1e-4) + cosine annealing schedule, mixed-precision (AMP) training, `cudnn.benchmark` for the fixed 224×224 input — tuned for a 6 GB RTX 3060 Laptop GPU.
5. After each epoch: validation loss + per-head accuracy appended to `curves.csv`.
6. **Early stopping**: if the monitored validation accuracy (grade accuracy for `grade`/`multi`, variety accuracy otherwise) doesn't improve for 7 epochs, training stops and the best checkpoint is restored.
7. Saves `best.pt` containing the state dict, full config, class names, and best validation accuracy.

### `evaluate.py` — held-out testing
Loads a run's `best.pt`, rebuilds the identical test split (same seed), and produces:
- `metrics.json` — accuracy, macro-F1, and per-class precision/recall/F1,
- `confusion_<head>.png` — confusion matrix figure,
- `probs_<head>.npy` + `labels_<head>.npy` — softmax outputs saved for the ensemble,
- `misclassified_<head>.csv` — file paths of every wrongly-classified test image, used for qualitative failure analysis.

### `ensemble.py` — ensemble learning (project Option 4)
Operates purely on the saved `probs_*.npy` files — no retraining. Computes:
- **Soft voting**: average the softmax probabilities of the selected runs, then argmax.
- **Weighted voting**: same, but each model's probabilities are weighted proportionally to its individual accuracy.
- **Disagreement analysis**: counts test images where the models were not unanimous, and how many of those the ensemble got right — the "why ensembling works" evidence (uncorrelated errors cancel). It also verifies all runs share identical test labels, guaranteeing a fair comparison.

### `plot_curves.py` — training/accuracy curves
Overlays the `curves.csv` of any set of runs onto shared loss and accuracy figures (the training-curve figures in the report).

### `run_all.ps1` — full reproduction
Runs all 13 experiments in sequence: 4 variety models, 4 grading models, 2 multi-head configurations, and 3 ablations (no-augmentation, no-dropout, frozen backbone), followed by evaluation, ensembling, and curve plotting.

---

## Setup

### 1. Environment

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
# Windows:            .venv\Scripts\activate
# Linux/macOS:        source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python ≥ 3.10 and a CUDA-capable GPU (all experiments ran on an RTX 3060 Laptop GPU, 6 GB VRAM). CPU works but is very slow.

### 2. Dataset

Download the **Date Fruit Dataset for Inspection and Grading** (Maitlo, Shaikh & Arain, Mendeley Data, DOI: [10.17632/s5zfvsw5kv.3](https://doi.org/10.17632/s5zfvsw5kv.3)) and extract it so that the folder structure is:

```
Date Fruit/
├── Aseel/<Size>/<Grade-N>/*.jpg
├── Fasli Toto/...
├── Gajar/...
└── Kupro/...
```

Only the **original** images folder is needed; the pre-augmented images are deliberately not used.

---

## Training (reproducing all experiments)

Reproduce everything with one script:

```powershell
cd src
.\run_all.ps1 -DataRoot "path\to\Date Fruit"
```

Or run any experiment individually. The 13 runs of the paper:

```bash
# ---- Variety classification (all 3,004 images, 4 classes) ----
python train.py --data-root "path/to/Date Fruit" --task variety --backbone scratch
python train.py --data-root "path/to/Date Fruit" --task variety --backbone resnet18
python train.py --data-root "path/to/Date Fruit" --task variety --backbone vgg16
python train.py --data-root "path/to/Date Fruit" --task variety --backbone googlenet

# ---- Quality grading (Gajar + Kupro only, 3 classes) ----
python train.py --data-root "path/to/Date Fruit" --task grade --backbone scratch
python train.py --data-root "path/to/Date Fruit" --task grade --backbone resnet18
python train.py --data-root "path/to/Date Fruit" --task grade --backbone vgg16
python train.py --data-root "path/to/Date Fruit" --task grade --backbone googlenet

# ---- Multi-task (shared backbone, variety + grade heads) ----
python train.py --data-root "path/to/Date Fruit" --task multi --backbone resnet18
python train.py --data-root "path/to/Date Fruit" --task multi --backbone resnet18 --grade-loss-weight 2.0

# ---- Ablations (each changes exactly one thing vs. the grade_resnet18 baseline) ----
python train.py --data-root "path/to/Date Fruit" --task grade --backbone resnet18 --no-aug
python train.py --data-root "path/to/Date Fruit" --task grade --backbone resnet18 --no-dropout
python train.py --data-root "path/to/Date Fruit" --task grade --backbone resnet18 --freeze
```

Each run creates `runs/<auto_generated_name>/` (e.g. `runs/grade_resnet18_noaug/`). All runs use `--seed 42` by default, so the train/val/test splits are identical everywhere.

---

## Testing / evaluation

Evaluate any trained run on the held-out test set (works with the released checkpoints — no training needed):

```bash
python evaluate.py --data-root "path/to/Date Fruit" --run-dir runs/grade_googlenet
```

Then build the ensembles from the saved softmax outputs:

```bash
# Grading ensemble (ResNet-18 + VGG-16 + GoogLeNet)
python ensemble.py --head grade   --runs runs/grade_resnet18 runs/grade_vgg16 runs/grade_googlenet

# Variety ensemble
python ensemble.py --head variety --runs runs/variety_resnet18 runs/variety_vgg16 runs/variety_googlenet
```

And regenerate the curve figures:

```bash
python plot_curves.py --runs runs/grade_resnet18 runs/grade_vgg16 runs/grade_googlenet runs/grade_scratch --out figures/
python plot_curves.py --runs runs/variety_resnet18 runs/variety_vgg16 runs/variety_googlenet runs/variety_scratch --out figures_variety/
python plot_curves.py --runs runs/grade_resnet18 runs/grade_resnet18_noaug runs/grade_resnet18_nodrop runs/grade_resnet18_frozen --out figures/
```

---

## Output files explained

Each `runs/<run_name>/` folder contains:

| File | Meaning |
|---|---|
| `config.json` | Exact hyperparameters/flags of the run (full reproducibility) |
| `curves.csv` | Per-epoch train loss, val loss, val accuracy per head, LR, wall-time |
| `best.pt` | Best-validation checkpoint (state dict + config + class names) — via Release |
| `metrics.json` | Test accuracy, macro-F1, per-class precision/recall/F1/support |
| `confusion_variety.png` / `confusion_grade.png` | Test-set confusion matrix figure(s) |
| `probs_<head>.npy`, `labels_<head>.npy` | Test softmax outputs + labels (consumed by `ensemble.py`) |
| `misclassified_<head>.csv` | Paths of misclassified test images (qualitative error analysis) |

Top-level result files:

| File | Meaning |
|---|---|
| `ensemble_grade.json` | Per-model + soft/weighted voting test metrics, voting weights, and the disagreement analysis (318 test images: 102 non-unanimous, 67 rescued by the ensemble) |
| `ensemble_variety.json` | Same for variety (451 test images: 46 non-unanimous, 44 rescued) |
| `figures/`, `figures_variety/` | Accuracy/loss curve figures used in the report |

---

## Full results

### Variety classification (4 classes, 451 test images)

| Model | Val. Acc. | Test Acc. | Test Macro-F1 |
|---|---|---|---|
| Scratch CNN | 95.34% | 95.12% | 0.9509 |
| ResNet-18 | 90.02% | 90.91% | 0.9103 |
| **VGG-16** | **99.11%** | **100.00%** | **1.0000** |
| GoogLeNet | 97.12% | 97.34% | 0.9721 |
| Soft / Weighted Voting | — | 99.56% | 0.9949 |

### Quality grading (3 classes, 318 test images)

| Model | Val. Acc. | Test Acc. | Test Macro-F1 |
|---|---|---|---|
| Scratch CNN | 76.73% | 72.96% | 0.7321 |
| ResNet-18 | 75.79% | 72.33% | 0.7332 |
| VGG-16 | 76.42% | 70.13% | 0.7117 |
| **GoogLeNet** | **80.19%** | **75.79%** | **0.7599** |
| Multi-head ResNet-18 | 75.79% | 73.27% | 0.7294 |
| Multi-head (λ = 2) | 75.16% | 72.33% | 0.7278 |
| **Soft / Weighted Voting** | — | **77.36%** | **0.7787** |

### Ablations (ResNet-18 grading baseline)

| Configuration | Val. Acc. | Test Acc. | Takeaway |
|---|---|---|---|
| Baseline | 75.79% | 72.33% | — |
| No augmentation | 77.99% | 75.79% | Slightly higher accuracy, but train loss → 0 while val loss rises above 1.2 → memorization / poor calibration |
| No dropout | 77.99% | 71.70% | Transfer learning + early stopping already regularize |
| Frozen backbone | 55.03% | 51.57% | **−20 pts: full fine-tuning is the single most important factor** |

---

## Reproducibility notes

- Every run uses `--seed 42` for splits and initialization; the identical test split across runs is what makes voting on saved probabilities valid (`ensemble.py` verifies this and raises an error otherwise).
- Mixed precision means bitwise results can vary marginally across GPU/driver versions; accuracies should reproduce to within a fraction of a percent.
- If the dataset lives on OneDrive/cloud-synced storage, the **first** epoch is slow due to file caching; later epochs reflect true training speed.

## References

Dataset: A. Maitlo, N. Shaikh, R. Arain, "Date Fruit Dataset for Inspection and Grading," Mendeley Data, v3, DOI: 10.17632/s5zfvsw5kv.3. See the project report for the full literature review.
