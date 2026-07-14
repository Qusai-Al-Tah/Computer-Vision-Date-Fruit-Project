"""
Evaluation on the held-out test split.

Produces per run:
- metrics.json: accuracy, per-class precision/recall/F1, macro-F1
  (11TrainingCNN error metrics)
- confusion_variety.png / confusion_grade.png
- probs_<head>.npy + labels_<head>.npy: softmax outputs saved for
  ensemble.py (soft/weighted voting over multiple runs)
- misclassified_<head>.csv: paths of wrongly classified test images for the
  qualitative failure analysis figure in the report

Uses the SAME seed as training so the test split is identical across runs.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (classification_report, confusion_matrix, f1_score)
from torch.utils.data import DataLoader

from dataset import prepare
from model import build_model


def plot_confusion(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--run-dir", required=True, help="Directory containing best.pt")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = prepare(args.data_root, cfg["task"], augment=False,
                   seed=cfg["seed"], img_size=cfg["img_size"])
    test_loader = DataLoader(
        data["test"], batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    dropout = 0.0 if cfg["no_dropout"] else cfg["dropout"]
    model = build_model(
        cfg["backbone"], cfg["task"],
        n_varieties=len(ckpt["variety_classes"]),
        n_grades=len(ckpt["grade_classes"]),
        dropout=dropout, freeze=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    heads = []
    if cfg["task"] in ("variety", "multi"):
        heads.append(("variety", ckpt["variety_classes"]))
    if cfg["task"] in ("grade", "multi"):
        heads.append(("grade", ckpt["grade_classes"]))

    probs = {h: [] for h, _ in heads}
    labels = {"variety": [], "grade": []}
    with torch.no_grad():
        for imgs, v_lab, g_lab in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(imgs)
            for h, _ in heads:
                probs[h].append(torch.softmax(out[h].float(), dim=1).cpu().numpy())
            labels["variety"].append(v_lab.numpy())
            labels["grade"].append(g_lab.numpy())

    metrics = {}
    test_paths = data["test"].df["path"].tolist()
    for h, classes in heads:
        P = np.concatenate(probs[h])
        y = np.concatenate(labels[h])
        pred = P.argmax(1)
        np.save(run_dir / f"probs_{h}.npy", P)
        np.save(run_dir / f"labels_{h}.npy", y)

        acc = float((pred == y).mean())
        macro_f1 = float(f1_score(y, pred, average="macro"))
        report = classification_report(
            y, pred, target_names=classes, output_dict=True, zero_division=0
        )
        metrics[h] = {"accuracy": acc, "macro_f1": macro_f1, "per_class": report}

        cm = confusion_matrix(y, pred)
        plot_confusion(cm, classes, f"{h} — {run_dir.name}",
                       run_dir / f"confusion_{h}.png")

        wrong = pred != y
        with open(run_dir / f"misclassified_{h}.csv", "w") as f:
            f.write("path,true,pred,confidence\n")
            for i in np.where(wrong)[0]:
                f.write(f"{test_paths[i]},{classes[y[i]]},{classes[pred[i]]},"
                        f"{P[i, pred[i]]:.3f}\n")

        print(f"[{run_dir.name}] {h}: acc={acc:.4f} macro_f1={macro_f1:.4f}")

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
