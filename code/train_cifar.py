"""CIFAR-10 ten-class experiment with a train-only validation split.

The official Test split is read once after validation chooses a checkpoint.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CKPT_DIR, CIFAR_MEAN, CIFAR_STD, evaluate, get_device,
                    save_metrics, seed_everything, train_model)
from models import SimpleCNN, count_params

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
SPLIT_SEED = 2026
DATA = os.environ.get("EE6483_DATA_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))


def split_indices(targets, seed=SPLIT_SEED):
    """Fixed stratified split: 4500 train / 500 validation per class."""
    targets = np.asarray(targets)
    rng = np.random.RandomState(seed)
    train_idx, val_idx = [], []
    for cls in range(10):
        indices = np.flatnonzero(targets == cls)
        if len(indices) != 5000:
            raise ValueError(f"CIFAR train class {cls} has {len(indices)} images")
        rng.shuffle(indices)
        val_idx.extend(indices[:500].tolist())
        train_idx.extend(indices[500:].tolist())
    return sorted(train_idx), sorted(val_idx)


def index_digest(indices):
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def datasets_and_indices(aug=True):
    train_ops = ([transforms.RandomCrop(32, padding=4),
                  transforms.RandomHorizontalFlip(p=0.5),
                  transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2)], p=0.5)]
                 if aug else [])
    train_tf = transforms.Compose(train_ops + [transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD), transforms.RandomErasing(p=0.25 if aug else 0)])
    eval_tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
    root = os.path.join(DATA, "cifar10")
    train_source = datasets.CIFAR10(root, train=True, transform=train_tf, download=False)
    val_source = datasets.CIFAR10(root, train=True, transform=eval_tf, download=False)
    test_source = datasets.CIFAR10(root, train=False, transform=eval_tf, download=False)
    train_idx, val_idx = split_indices(train_source.targets)
    if set(train_idx) & set(val_idx):
        raise AssertionError("train/validation overlap")
    return train_source, val_source, test_source, train_idx, val_idx


def loaders(batch_size=128, workers=4, *, smoke=False):
    train_source, val_source, test_source, train_idx, val_idx = datasets_and_indices()
    if smoke:
        targets = np.asarray(train_source.targets)
        train_idx = [i for cls in range(10) for i in [j for j in train_idx if targets[j] == cls][:20]]
        val_idx = [i for cls in range(10) for i in [j for j in val_idx if targets[j] == cls][:10]]
        test_source = Subset(test_source, range(100))
    train_ds = Subset(train_source, train_idx)
    val_ds = Subset(val_source, val_idx)
    options = dict(num_workers=workers, persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **options)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, **options)
    test_loader = DataLoader(test_source, batch_size=512, shuffle=False, **options)
    return train_ds, val_ds, test_source, train_loader, val_loader, test_loader, train_idx, val_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
    seed_everything(args.seed)
    device = get_device()
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, train_idx, val_idx = loaders(
        args.batch_size, args.workers, smoke=args.smoke)
    model = SimpleCNN(num_classes=10, base=64, dropout=0.5, cifar=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    run_name = f"cifar10_s{args.seed}" + ("_smoke" if args.smoke else "")
    ckpt = os.path.join(CKPT_DIR, f"{run_name}.pt")
    print(f"device={device} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)
    history, best_val = train_model(model, train_loader, val_loader, device,
        epochs=args.epochs, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, tag=run_name, save_ckpt=ckpt, log_every=0)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device)
    _, test_acc, cm, _ = evaluate(model, test_loader, device)
    per_class = (np.diag(cm) / cm.sum(1)).tolist()
    save_metrics(run_name, dict(seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, wd=args.wd,
        label_smoothing=args.label_smoothing, n_params=count_params(model),
        n_train=len(train_ds), n_val=len(val_ds), n_test=len(test_ds),
        split_seed=SPLIT_SEED, train_index_sha256=index_digest(train_idx),
        val_index_sha256=index_digest(val_idx), history=history,
        best_val_acc=best_val, best_epoch=int(np.argmax(history['val_acc']) + 1),
        final_test_acc=test_acc, confusion_matrix=cm.tolist(),
        per_class_acc=per_class, classes=CLASSES,
        selection_split="CIFAR-10 official train, stratified holdout",
        test_usage="single final evaluation after checkpoint selection"))
    print(f"[RESULT] {run_name}: val={best_val:.4f} test={test_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
