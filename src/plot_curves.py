"""
Plot training/validation curves from one or more runs onto shared figures
(the 'training curves, accuracy curves' the report requires).

Usage:
  python plot_curves.py --runs runs/grade_resnet18 runs/grade_resnet18_noaug --out figures/
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--head", choices=["variety", "grade"], default="grade")
    p.add_argument("--out", default="figures")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(6, 4))   # loss
    fig2, ax2 = plt.subplots(figsize=(6, 4))   # val accuracy

    for r in args.runs:
        r = Path(r)
        df = pd.read_csv(r / "curves.csv")
        ax1.plot(df["epoch"], df["train_loss"], label=f"{r.name} (train)")
        ax1.plot(df["epoch"], df["val_loss"], "--", label=f"{r.name} (val)")
        ax2.plot(df["epoch"], df[f"val_acc_{args.head}"], label=r.name)

    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training / validation loss"); ax1.legend(fontsize=7)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel(f"Val accuracy ({args.head})")
    ax2.set_title("Validation accuracy"); ax2.legend(fontsize=7)
    for fig, name in ((fig1, "loss_curves.png"), (fig2, "acc_curves.png")):
        fig.tight_layout()
        fig.savefig(out / name, dpi=150)
    print(f"Saved figures to {out}/")


if __name__ == "__main__":
    main()
