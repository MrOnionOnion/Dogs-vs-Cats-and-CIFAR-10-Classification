"""Shared utilities: seeding, device selection, training loop, evaluation."""
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(ROOT, "outputs", "metrics")
CKPT_DIR = os.path.join(ROOT, "outputs", "ckpts")
FIG_DIR = os.path.join(ROOT, "outputs", "figures")
SAMPLE_DIR = os.path.join(ROOT, "outputs", "samples")


def seed_everything(seed: int):
    """Fix every RNG we control. MPS conv kernels remain non-deterministic,
    which is why the headline model is averaged over 3 seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.empty(1, device="mps")
            return torch.device("mps")
        except RuntimeError:
            pass
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    """Return (loss, accuracy, confusion_matrix KxK numpy, per_sample list)."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    k = getattr(model, "num_classes", 2)
    cm = np.zeros((k, k), dtype=int)
    samples = []  # (index, true, pred, confidence, prob_pred)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * y.size(0)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        for i in range(y.size(0)):
            cm[y[i].item(), pred[i].item()] += 1
            samples.append((y[i].item(), pred[i].item(),
                            conf[i].item(), probs[i].tolist()))
    loss = loss_sum / max(total, 1)
    return loss, correct / total, cm, samples


def train_model(model, train_loader, val_loader, device, *, epochs, optimizer,
                scheduler=None, criterion=None, log_every=50, tag="run",
                save_ckpt=None, freeze_epochs=0, freeze_params=None):
    """Generic training loop. If freeze_epochs > 0, `freeze_params` are frozen
    for the first N epochs (linear probing) then unfrozen."""
    criterion = criterion or nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "epoch_time": [], "lr": []}
    best_val_acc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        t0 = time.time()
        if freeze_epochs and freeze_params is not None:
            frozen = ep <= freeze_epochs
            for p in freeze_params:
                p.requires_grad_(not frozen)
            model.train()
        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        for it, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * y.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()
            run_total += y.size(0)
            if log_every and (it + 1) % log_every == 0:
                print(f"[{tag}] ep{ep} it{it+1}/{len(train_loader)} "
                      f"loss={run_loss/run_total:.4f} acc={run_correct/run_total:.4f}",
                      flush=True)
        if scheduler is not None:
            scheduler.step()
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device, criterion)
        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(run_loss / run_total)
        history["train_acc"].append(run_correct / run_total)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(dt)
        history["lr"].append(lr_now)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        print(f"[{tag}] epoch {ep}/{epochs} train_loss={run_loss/run_total:.4f} "
              f"train_acc={run_correct/run_total:.4f} val_loss={val_loss:.4f} "
              f"val_acc={val_acc:.4f} ({dt:.0f}s) lr={lr_now:.2e}", flush=True)
    if save_ckpt and best_state is not None:
        os.makedirs(os.path.dirname(save_ckpt), exist_ok=True)
        torch.save(best_state, save_ckpt)
    return history, best_val_acc


def save_metrics(name, payload):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {path}")
    return path
