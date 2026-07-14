"""
Dataset loader for the Date Fruit Dataset for Inspection and Grading
(Maitlo, Shaikh, Arain — Mendeley Data, DOI: 10.17632/s5zfvsw5kv.3).

Directory structure expected:
    <root>/<Variety>/<Size>/<Grade-N>/*.jpg

Key design decisions (see report, Data section):
- Only ORIGINAL images are used; the authors' pre-augmented folder is excluded
  to prevent train/test leakage. Augmentation is applied on-the-fly to the
  training split only (rotation, flipping, scaling — per course slides on
  regularization via data augmentation).
- For the GRADING task, only varieties with more than one grade present
  (Gajar, Kupro) are used. Aseel and Fasli Toto contain exclusively Grade-1
  images, which would let a grade classifier shortcut through variety
  appearance and inflate accuracy.
- Splits are stratified by the (variety, size, grade) cell so every populated
  cell is represented proportionally in train/val/test (70/15/15).
"""

import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_index(root: str) -> pd.DataFrame:
    """Walk root and return a DataFrame with columns:
    path, variety, size, grade, cell."""
    rows = []
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    for variety_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for size_dir in sorted(p for p in variety_dir.iterdir() if p.is_dir()):
            for grade_dir in sorted(p for p in size_dir.iterdir() if p.is_dir()):
                for f in grade_dir.iterdir():
                    if f.suffix.lower() in IMG_EXTS:
                        rows.append(
                            {
                                "path": str(f),
                                "variety": variety_dir.name.strip(),
                                "size": size_dir.name.strip(),
                                "grade": grade_dir.name.strip(),
                            }
                        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            f"No images found under {root}. Check the path points at the "
            "'Date Fruit' folder containing the variety subfolders."
        )
    df["cell"] = df["variety"] + "|" + df["size"] + "|" + df["grade"]
    return df


def filter_for_task(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """For grading, keep only varieties that have >1 distinct grade."""
    if task in ("grade", "multi"):
        grades_per_variety = df.groupby("variety")["grade"].nunique()
        gradable = grades_per_variety[grades_per_variety > 1].index.tolist()
        df = df[df["variety"].isin(gradable)].reset_index(drop=True)
    return df


def make_splits(df: pd.DataFrame, seed: int = 42):
    """Stratified 70/15/15 split by (variety,size,grade) cell."""
    trainval, test = train_test_split(
        df, test_size=0.15, stratify=df["cell"], random_state=seed
    )
    train, val = train_test_split(
        trainval,
        test_size=0.15 / 0.85,
        stratify=trainval["cell"],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def get_transforms(augment: bool, img_size: int = 224):
    """Training transform (optionally augmented) — augmentation limited to
    rotation, flipping, scaling per the course slides."""
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if augment:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),  # scaling
                transforms.RandomHorizontalFlip(),                          # flipping
                transforms.RandomVerticalFlip(),   # dates have no canonical 'up'
                transforms.RandomRotation(30),                              # rotation
                transforms.ToTensor(),
                norm,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            norm,
        ]
    )


class DateFruitDataset(Dataset):
    """Returns (image, variety_idx, grade_idx). Single-task training just
    ignores the label it doesn't need."""

    def __init__(self, df: pd.DataFrame, transform, variety_classes, grade_classes):
        self.df = df
        self.transform = transform
        self.variety_to_idx = {v: i for i, v in enumerate(variety_classes)}
        self.grade_to_idx = {g: i for i, g in enumerate(grade_classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        img = self.transform(img)
        return (
            img,
            self.variety_to_idx[row["variety"]],
            self.grade_to_idx[row["grade"]],
        )


def class_weights(df: pd.DataFrame, column: str, classes) -> torch.Tensor:
    """Inverse-frequency weights for weighted cross-entropy on imbalanced
    classes (metrics/imbalance handling per course slides)."""
    counts = df[column].value_counts()
    w = torch.tensor(
        [len(df) / (len(classes) * counts[c]) for c in classes], dtype=torch.float32
    )
    return w


def prepare(root: str, task: str, augment: bool, seed: int = 42, img_size: int = 224):
    """One-stop: returns datasets, class lists, weights, and the count table."""
    df = build_index(root)
    count_table = (
        df.pivot_table(
            index="variety", columns="grade", values="path", aggfunc="count"
        )
        .fillna(0)
        .astype(int)
    )
    df = filter_for_task(df, task)
    variety_classes = sorted(df["variety"].unique())
    grade_classes = sorted(df["grade"].unique())

    train_df, val_df, test_df = make_splits(df, seed)

    train_ds = DateFruitDataset(
        train_df, get_transforms(augment, img_size), variety_classes, grade_classes
    )
    eval_tf = get_transforms(False, img_size)
    val_ds = DateFruitDataset(val_df, eval_tf, variety_classes, grade_classes)
    test_ds = DateFruitDataset(test_df, eval_tf, variety_classes, grade_classes)

    weights = {
        "variety": class_weights(train_df, "variety", variety_classes),
        "grade": class_weights(train_df, "grade", grade_classes),
    }
    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "variety_classes": variety_classes,
        "grade_classes": grade_classes,
        "weights": weights,
        "count_table": count_table,
        "split_sizes": (len(train_df), len(val_df), len(test_df)),
    }
