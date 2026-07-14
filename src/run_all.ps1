# ============================================================
# run_all.ps1 — full overnight experiment grid (RTX 3060)
# Run from the src/ folder:  .\run_all.ps1
# Total: 13 training runs. Rough estimate 4-7 hours on a 3060.
# ============================================================

$DATA = "C:\Users\qusai\OneDrive\Desktop\Date Fruit"

# ---------- Task 1: variety classification (all 4 varieties) ----------
python train.py --data-root $DATA --task variety --backbone scratch   --epochs 40
python train.py --data-root $DATA --task variety --backbone resnet18  --epochs 30
python train.py --data-root $DATA --task variety --backbone vgg16     --epochs 30 --batch-size 16
python train.py --data-root $DATA --task variety --backbone googlenet --epochs 30

# ---------- Task 2: grading (Gajar + Kupro only) ----------
python train.py --data-root $DATA --task grade --backbone scratch   --epochs 40
python train.py --data-root $DATA --task grade --backbone resnet18  --epochs 30
python train.py --data-root $DATA --task grade --backbone vgg16     --epochs 30 --batch-size 16
python train.py --data-root $DATA --task grade --backbone googlenet --epochs 30

# ---------- Multi-head: does variety knowledge help grading? ----------
python train.py --data-root $DATA --task multi --backbone resnet18 --epochs 30
python train.py --data-root $DATA --task multi --backbone resnet18 --epochs 30 --grade-loss-weight 2.0

# ---------- Ablations (on the grading task, resnet18) ----------
python train.py --data-root $DATA --task grade --backbone resnet18 --epochs 30 --no-aug
python train.py --data-root $DATA --task grade --backbone resnet18 --epochs 30 --no-dropout
python train.py --data-root $DATA --task grade --backbone resnet18 --epochs 30 --freeze

# ---------- Evaluate everything ----------
Get-ChildItem runs -Directory | ForEach-Object {
    python evaluate.py --data-root $DATA --run-dir $_.FullName
}

# ---------- Ensembles (soft + weighted voting over the three backbones) ----------
python ensemble.py --head variety --runs runs/variety_resnet18 runs/variety_vgg16 runs/variety_googlenet --out ensemble_variety.json
python ensemble.py --head grade   --runs runs/grade_resnet18 runs/grade_vgg16 runs/grade_googlenet --out ensemble_grade.json

# ---------- Figures ----------
python plot_curves.py --head grade   --runs runs/grade_resnet18 runs/grade_vgg16 runs/grade_googlenet runs/grade_scratch --out figures
python plot_curves.py --head grade   --runs runs/grade_resnet18 runs/grade_resnet18_noaug runs/grade_resnet18_nodrop runs/grade_resnet18_frozen --out figures_ablation
python plot_curves.py --head variety --runs runs/variety_resnet18 runs/variety_vgg16 runs/variety_googlenet runs/variety_scratch --out figures_variety

Write-Host "ALL DONE. Check runs/*/metrics.json, ensemble_*.json, figures*/"
