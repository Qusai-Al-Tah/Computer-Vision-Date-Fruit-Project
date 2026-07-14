"""
Training script. Every ablation is a command-line flag so the whole grid is
just a list of commands (see run_all.ps1 / run_all.sh).

GPU efficiency (RTX 3060):
- Mixed precision (torch.amp) — roughly halves memory and speeds up training
- cudnn.benchmark = True — autotunes conv kernels for fixed input size
- Parallel data loading (num_workers, pin_memory, persistent_workers)

Course-slide mapping:
- Early stopping, dropout, augmentation, L2 (weight_decay): 11TrainingCNN
  regularization
- Weighted cross-entropy + macro metrics: 11TrainingCNN error metrics
- Fine-tuning pretrained backbones: 12FamousCNN
"""

import argparse
import copy
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import prepare
from model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Path to the 'Date Fruit' folder")
    p.add_argument("--task", choices=["variety", "grade", "multi"], default="grade")
    p.add_argument("--backbone", default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50", "vgg16",
                            "googlenet", "scratch"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--no-aug", action="store_true", help="Disable augmentation (ablation)")
    p.add_argument("--no-dropout", action="store_true", help="Disable dropout (ablation)")
    p.add_argument("--freeze", action="store_true", help="Freeze backbone (ablation)")
    p.add_argument("--no-weighted-loss", action="store_true")
    p.add_argument("--grade-loss-weight", type=float, default=1.0,
                   help="Weight of grade loss vs variety loss in multi-task")
    p.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-name", default=None)
    p.add_argument("--out-dir", default="runs")
    return p.parse_args()


def evaluate_epoch(model, loader, device, task, criteria):
    model.eval()
    totals = {"variety": [0, 0], "grade": [0, 0]}  # correct, count
    loss_sum, n_batches = 0.0, 0
    with torch.no_grad():
        for imgs, v_lab, g_lab in loader:
            imgs = imgs.to(device, non_blocking=True)
            v_lab = v_lab.to(device, non_blocking=True)
            g_lab = g_lab.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(imgs)
                loss = 0.0
                if "variety" in out:
                    loss = loss + criteria["variety"](out["variety"], v_lab)
                    totals["variety"][0] += (out["variety"].argmax(1) == v_lab).sum().item()
                    totals["variety"][1] += len(v_lab)
                if "grade" in out:
                    loss = loss + criteria["grade"](out["grade"], g_lab)
                    totals["grade"][0] += (out["grade"].argmax(1) == g_lab).sum().item()
                    totals["grade"][1] += len(g_lab)
            loss_sum += loss.detach().item() if hasattr(loss, "detach") else float(loss)
            n_batches += 1
    accs = {k: (c / n if n else 0.0) for k, (c, n) in totals.items()}
    return loss_sum / max(n_batches, 1), accs


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available, training on CPU will be very slow.")

    augment = not args.no_aug
    dropout = 0.0 if args.no_dropout else args.dropout

    data = prepare(args.data_root, args.task, augment,
                   seed=args.seed, img_size=args.img_size)
    print("Variety x Grade counts (full dataset):")
    print(data["count_table"])
    print(f"Split sizes (train/val/test): {data['split_sizes']}")
    print(f"Task '{args.task}' classes -> varieties: {data['variety_classes']}, "
          f"grades: {data['grade_classes']}")

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(data["train"], shuffle=True, drop_last=False, **loader_kw)
    val_loader = DataLoader(data["val"], shuffle=False, **loader_kw)

    model = build_model(
        args.backbone, args.task,
        n_varieties=len(data["variety_classes"]),
        n_grades=len(data["grade_classes"]),
        dropout=dropout, freeze=args.freeze,
    ).to(device)

    criteria = {}
    for head in ("variety", "grade"):
        w = None if args.no_weighted_loss else data["weights"][head].to(device)
        criteria[head] = nn.CrossEntropyLoss(weight=w)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    run_name = args.run_name or (
        f"{args.task}_{args.backbone}"
        f"{'_noaug' if args.no_aug else ''}"
        f"{'_nodrop' if args.no_dropout else ''}"
        f"{'_frozen' if args.freeze else ''}"
        f"{'_glw' + str(args.grade_loss_weight) if args.task == 'multi' and args.grade_loss_weight != 1.0 else ''}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    curves_path = out_dir / "curves.csv"
    with open(curves_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "val_acc_variety", "val_acc_grade", "lr", "seconds"]
        )

    monitor = "grade" if args.task in ("grade", "multi") else "variety"
    best_acc, best_state, epochs_no_improve = 0.0, None, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, n_batches = time.time(), 0.0, 0
        for imgs, v_lab, g_lab in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            v_lab = v_lab.to(device, non_blocking=True)
            g_lab = g_lab.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(imgs)
                loss = 0.0
                if "variety" in out:
                    loss = loss + criteria["variety"](out["variety"], v_lab)
                if "grade" in out:
                    loss = loss + args.grade_loss_weight * criteria["grade"](out["grade"], g_lab)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += loss.detach().item() if hasattr(loss, "detach") else float(loss)
            n_batches += 1
        scheduler.step()

        val_loss, val_accs = evaluate_epoch(model, val_loader, device, args.task, criteria)
        seconds = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"[{run_name}] epoch {epoch:02d}/{args.epochs} "
            f"train_loss={loss_sum / n_batches:.4f} val_loss={val_loss:.4f} "
            f"val_acc_variety={val_accs['variety']:.4f} val_acc_grade={val_accs['grade']:.4f} "
            f"({seconds:.0f}s)"
        )
        with open(curves_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, loss_sum / n_batches, val_loss,
                 val_accs["variety"], val_accs["grade"], lr_now, round(seconds, 1)]
            )

        # Early stopping (11TrainingCNN: regularization)
        if val_accs[monitor] > best_acc:
            best_acc = val_accs[monitor]
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no val improvement "
                      f"for {args.patience} epochs).")
                break

    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(args),
            "variety_classes": data["variety_classes"],
            "grade_classes": data["grade_classes"],
            "best_val_acc": best_acc,
        },
        out_dir / "best.pt",
    )
    print(f"Done. Best val {monitor} acc = {best_acc:.4f}. Saved to {out_dir}/best.pt")


if __name__ == "__main__":
    main()
