"""CIFAR-10 step imbalance with a train-only validation split.

Same architecture, train indices, validation set, test set and update budget
for plain CE, class-weighted CE and weighted sampling.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CKPT_DIR, evaluate, get_device, save_metrics, seed_everything, train_model
from models import SimpleCNN, count_params
from train_cifar import CLASSES, SPLIT_SEED, datasets_and_indices, index_digest

MINORITY = [3, 5, 7]
IMB_SEED = 42


def imbalanced_indices(train_idx, targets, minority_count=450):
    targets = np.asarray(targets)
    rng = np.random.RandomState(IMB_SEED)
    keep = []
    for cls in range(10):
        cls_idx = np.array([i for i in train_idx if targets[i] == cls])
        rng.shuffle(cls_idx)
        n = minority_count if cls in MINORITY else len(cls_idx)
        keep.extend(cls_idx[:n].tolist())
    keep.sort()
    counts = np.bincount(targets[keep], minlength=10)
    return keep, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["baseline", "weighted", "oversamp"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
    seed_everything(args.seed)
    device = get_device()
    train_source, val_source, test_source, train_idx, val_idx = datasets_and_indices()
    targets = np.asarray(train_source.targets)
    if args.smoke:
        train_idx = [i for cls in range(10) for i in [j for j in train_idx if targets[j] == cls][:20]]
        val_idx = [i for cls in range(10) for i in [j for j in val_idx if targets[j] == cls][:10]]
        test_source = Subset(test_source, range(100))
    keep_idx, counts = imbalanced_indices(train_idx, targets, minority_count=2 if args.smoke else 450)
    train_ds = Subset(train_source, keep_idx)
    val_ds = Subset(val_source, val_idx)
    options = dict(num_workers=args.workers, persistent_workers=args.workers > 0)
    if args.method == "oversamp":
        sample_weights = torch.as_tensor(1.0 / counts[targets[keep_idx]], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, **options)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **options)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, **options)
    test_loader = DataLoader(test_source, batch_size=512, shuffle=False, **options)
    if args.method == "weighted":
        weights = torch.as_tensor(counts.sum() / (10.0 * counts), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model = SimpleCNN(num_classes=10, base=64, dropout=0.5, cifar=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    run_name = f"cifar10imb_{args.method}" + ("_smoke" if args.smoke else "")
    ckpt = os.path.join(CKPT_DIR, f"{run_name}.pt")
    print(f"device={device} train_counts={counts.tolist()} val={len(val_ds)} test={len(test_source)}", flush=True)
    history, best_val = train_model(model, train_loader, val_loader, device,
        epochs=args.epochs, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, tag=run_name, save_ckpt=ckpt, log_every=0)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device)
    _, test_acc, cm, _ = evaluate(model, test_loader, device)
    recall = np.diag(cm) / cm.sum(1)
    save_metrics(run_name, dict(method=args.method, seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, weight_decay=5e-4,
        label_smoothing=0.1, n_params=count_params(model),
        n_train=len(train_ds), n_val=len(val_ds), n_test=len(test_source),
        train_class_counts=counts.tolist(), minority_classes=MINORITY,
        split_seed=SPLIT_SEED, imbalance_seed=IMB_SEED,
        train_index_sha256=index_digest(keep_idx), val_index_sha256=index_digest(val_idx),
        history=history, best_val_acc=best_val,
        best_epoch=int(np.argmax(history['val_acc']) + 1), test_acc=test_acc,
        confusion_matrix=cm.tolist(), per_class_recall=recall.tolist(),
        macro_recall=float(recall.mean()), minority_recall=float(recall[MINORITY].mean()),
        classes=CLASSES,
        selection_split="CIFAR-10 official train, stratified holdout",
        test_usage="single final evaluation after checkpoint selection"))
    print(f"[RESULT] {run_name}: val={best_val:.4f} test={test_acc:.4f} "
          f"minority_recall={recall[MINORITY].mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
