"""
Ensemble learning (project Option 4): combine the saved softmax outputs of
several already-trained models. No retraining needed.

- Soft voting: average the softmax probabilities
- Weighted voting: weight each model by its own test accuracy
- Disagreement analysis: cases where individual models disagreed and the
  ensemble was right (report figure: why ensembling helps — uncorrelated
  errors cancel)

Usage:
  python ensemble.py --head grade --runs runs/grade_resnet18 runs/grade_vgg16 runs/grade_googlenet
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", choices=["variety", "grade"], required=True)
    p.add_argument("--runs", nargs="+", required=True,
                   help="Run directories containing probs_<head>.npy")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    probs, names = [], []
    y = None
    for r in args.runs:
        r = Path(r)
        P = np.load(r / f"probs_{args.head}.npy")
        labels = np.load(r / f"labels_{args.head}.npy")
        if y is None:
            y = labels
        elif not np.array_equal(y, labels):
            raise RuntimeError(
                f"{r} has different test labels — all runs must use the same "
                "--seed so the test split is identical."
            )
        probs.append(P)
        names.append(r.name)
    probs = np.stack(probs)  # (n_models, n_samples, n_classes)

    results = {}
    ind_accs = []
    for i, name in enumerate(names):
        pred = probs[i].argmax(1)
        acc = float((pred == y).mean())
        ind_accs.append(acc)
        results[name] = {
            "accuracy": acc,
            "macro_f1": float(f1_score(y, pred, average="macro")),
        }

    # Soft voting
    soft_pred = probs.mean(0).argmax(1)
    results["ensemble_soft_voting"] = {
        "accuracy": float((soft_pred == y).mean()),
        "macro_f1": float(f1_score(y, soft_pred, average="macro")),
    }

    # Weighted voting (weights = individual accuracies, normalized)
    w = np.array(ind_accs)
    w = w / w.sum()
    weighted_pred = (probs * w[:, None, None]).sum(0).argmax(1)
    results["ensemble_weighted_voting"] = {
        "accuracy": float((weighted_pred == y).mean()),
        "macro_f1": float(f1_score(y, weighted_pred, average="macro")),
        "weights": {n: float(x) for n, x in zip(names, w)},
    }

    # Disagreement analysis
    ind_preds = probs.argmax(2)                      # (n_models, n_samples)
    disagree = (ind_preds != ind_preds[0]).any(0)
    saved = disagree & (soft_pred == y) & (ind_preds != y).any(0)
    results["disagreement_analysis"] = {
        "n_test": int(len(y)),
        "n_models_disagree": int(disagree.sum()),
        "n_ensemble_correct_where_some_model_wrong": int(saved.sum()),
        "indices_saved_by_ensemble": np.where(saved)[0].tolist()[:50],
    }

    print(f"\n=== Ensemble results: head = {args.head} ===")
    for k, v in results.items():
        if "accuracy" in v:
            print(f"{k:32s} acc={v['accuracy']:.4f} macro_f1={v['macro_f1']:.4f}")
    d = results["disagreement_analysis"]
    print(f"\nModels disagreed on {d['n_models_disagree']}/{d['n_test']} test images; "
          f"ensemble corrected {d['n_ensemble_correct_where_some_model_wrong']} of them.")

    out = args.out or f"ensemble_{args.head}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
