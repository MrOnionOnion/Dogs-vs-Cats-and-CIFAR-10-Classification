"""Dogs-vs-Cats experiments.

Modes
-----
main     : SimpleCNN-DVC @128px + augmentation (headline model, run with 3 seeds)
noaug    : same but WITHOUT augmentation (ablation)
small64  : same but @64px input (resolution ablation)
resnet   : fine-tuned ImageNet ResNet-18 @160px (transfer learning)
lrsweep  : quick learning-rate sweep on a 2000-image subset

Usage: python3 code/train_dvc.py --mode main --seed 0
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CKPT_DIR, IMAGENET_MEAN, IMAGENET_STD, METRICS_DIR,
                    evaluate, get_device, save_metrics, seed_everything,
                    train_model)
from models import SimpleCNN, build_resnet18, count_params

DATA = os.environ.get("EE6483_DATA_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))


def train_transform(size, aug=True):
    t = []
    if aug:
        t += [
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.8),
            transforms.RandomRotation(10),
        ]
    else:
        t += [transforms.Resize(size + 16), transforms.CenterCrop(size)]
    t += [transforms.ToTensor(),
          transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(t)


def eval_transform(size):
    return transforms.Compose([
        transforms.Resize(size + 16),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def balanced_subset(dataset, per_class, rng_seed):
    """Deterministically pick per_class images per class from an ImageFolder."""
    targets = np.array(dataset.targets)
    rng = np.random.RandomState(rng_seed)
    idx = []
    for c in np.unique(targets):
        cand = np.where(targets == c)[0]
        rng.shuffle(cand)
        idx.extend(cand[:per_class].tolist())
    idx = np.sort(np.array(idx))
    return Subset(dataset, idx.tolist())


def make_loaders(size, per_class_train, batch_size, workers, aug=True):
    train_ds = datasets.ImageFolder(os.path.join(DATA, "train"), transform=train_transform(size, aug))
    val_ds = datasets.ImageFolder(os.path.join(DATA, "val"), transform=eval_transform(size))
    if per_class_train > 0:
        train_ds = balanced_subset(train_ds, per_class_train, rng_seed=123)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=workers, persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=workers, persistent_workers=workers > 0)
    return train_ds, val_ds, train_loader, val_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="main",
                    choices=["main", "noaug", "small64", "resnet", "lrsweep"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--smoke", action="store_true",
                    help="40 training images, one epoch; pipeline check only")
    args = ap.parse_args()

    device = get_device()
    print(f"device={device} mode={args.mode} seed={args.seed}")

    if args.mode == "lrsweep":
        return run_lrsweep(args, device)

    cfg = {
        "main":    dict(size=128, arch="cnn", aug=True,  epochs=18, per_class=5000, lr=1e-3, wd=1e-4),
        "noaug":   dict(size=128, arch="cnn", aug=False, epochs=18, per_class=5000, lr=1e-3, wd=1e-4),
        "small64": dict(size=64,  arch="cnn", aug=True,  epochs=18, per_class=5000, lr=1e-3, wd=1e-4),
        "resnet":  dict(size=160, arch="resnet18", aug=True, epochs=8, per_class=5000, lr=1e-3, wd=1e-4),
    }[args.mode]
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.smoke:
        cfg["epochs"] = 1
        cfg["per_class"] = 20

    seed_everything(args.seed)
    train_ds, val_ds, train_loader, val_loader = make_loaders(
        cfg["size"], cfg["per_class"], args.batch_size, args.workers, cfg["aug"])

    if cfg["arch"] == "cnn":
        model = SimpleCNN(num_classes=2, base=32, dropout=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                     weight_decay=cfg["wd"])
        freeze_params, freeze_epochs = None, 0
    else:
        model = build_resnet18(num_classes=2, pretrained=True).to(device)
        head_ids = [id(p) for p in model.fc.parameters()]
        backbone = [p for p in model.parameters() if id(p) not in head_ids]
        optimizer = torch.optim.AdamW(
            [{"params": backbone, "lr": cfg["lr"] / 10},
             {"params": model.fc.parameters(), "lr": cfg["lr"]}],
            weight_decay=cfg["wd"])
        freeze_params, freeze_epochs = backbone, 2  # linear probe then fine-tune

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    run_name = f"dvc_{args.mode}" + (f"_s{args.seed}" if args.mode == "main" else "")
    if args.smoke:
        run_name += "_smoke"
    ckpt = os.path.join(CKPT_DIR, f"{run_name}.pt")

    n_params = count_params(model)
    print(f"{run_name}: {n_params/1e6:.2f}M trainable params, "
          f"{len(train_ds)} train / {len(val_ds)} val images, input {cfg['size']}px")

    history, best_val = train_model(
        model, train_loader, val_loader, device,
        epochs=cfg["epochs"], optimizer=optimizer, scheduler=sched,
        tag=run_name, save_ckpt=ckpt,
        freeze_epochs=freeze_epochs, freeze_params=freeze_params)

    # final evaluation with the best checkpoint
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device)
    val_loss, val_acc, cm, _ = evaluate(model, val_loader, device,
                                        nn.CrossEntropyLoss())
    save_metrics(run_name, dict(
        mode=args.mode, seed=args.seed, arch=cfg["arch"], size=cfg["size"],
        aug=cfg["aug"], epochs=cfg["epochs"], batch_size=args.batch_size,
        lr=cfg["lr"], weight_decay=cfg["wd"], n_params=n_params,
        n_train=len(train_ds), n_val=len(val_ds), history=history,
        best_val_acc=best_val, final_val_acc=val_acc,
        confusion_matrix=cm.tolist(), class_names=train_ds.dataset.classes
        if cfg["per_class"] > 0 else train_ds.classes))
    print(f"[RESULT] {run_name}: best_val_acc={best_val:.4f} final_val_acc={val_acc:.4f}")


def run_lrsweep(args, device):
    """3-epoch runs on a 2000-image subset (1000/class) to pick the LR."""
    lrs = [1e-4, 3e-4, 1e-3, 3e-3]
    results = []
    for lr in lrs:
        seed_everything(args.seed)
        train_ds, val_ds, train_loader, val_loader = make_loaders(
            128, 1000, args.batch_size, args.workers, aug=True)
        model = SimpleCNN(num_classes=2, base=32, dropout=0.5).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        hist, best = train_model(model, train_loader, val_loader, device,
                                 epochs=3, optimizer=opt, tag=f"lrsweep-{lr:.0e}",
                                 log_every=0)
        results.append(dict(lr=lr, best_val_acc=best,
                            final_val_acc=hist["val_acc"][-1]))
        print(f"[lrsweep] lr={lr:.0e} best={best:.4f}")
    save_metrics("dvc_lrsweep", dict(seed=args.seed, size=128, arch="cnn",
        aug=True, epochs=3, batch_size=args.batch_size, weight_decay=1e-4,
        subset_seed=123, n_train=2000, n_val=5000, results=results))
    best_row = max(results, key=lambda r: r["best_val_acc"])
    print(f"[lrsweep] winner: lr={best_row['lr']:.0e} ({best_row['best_val_acc']:.4f})")


if __name__ == "__main__":
    main()
