"""Predict the Dogs-vs-Cats test set -> submission.csv, and export sample
grids for the qualitative analysis (question e).

This finalizer uses ResNet-18 at 160 px and the 128 px SimpleCNN main runs.
Mixed ensembles are evaluated per-model and softmax probabilities averaged.
The 64 px ablation checkpoint is evaluated by verify_dvc_cnn_ckpts.py.

Usage:
  python3 code/predict_test.py --model outputs/ckpts/dvc_resnet.pt --tag resnet
  python3 code/predict_test.py --ensemble outputs/ckpts/dvc_main_s0.pt,... --tag ens
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (IMAGENET_MEAN, IMAGENET_STD, SAMPLE_DIR, get_device,
                    seed_everything)
from models import SimpleCNN, build_resnet18

DATA = os.environ.get("EE6483_DATA_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))
IDX2CLS = {0: "cat", 1: "dog"}


class FlatImageDataset(torch.utils.data.Dataset):
    """The test folder holds flat N.jpg files with no class subdirectories,
    so torchvision's ImageFolder cannot be used there."""

    def __init__(self, root, transform):
        self.files = sorted(
            (os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(".jpg")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.files[i]).convert("RGB")
        return self.transform(img), 0


def eval_tf(size):
    return transforms.Compose([
        transforms.Resize(size + 16), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "fc.weight" in ck:
        model = build_resnet18(num_classes=2, pretrained=False)
        size = 160
    else:
        model = SimpleCNN(num_classes=2, base=32, dropout=0.5)
        size = 128
    model.load_state_dict(ck)
    model.to(device).eval()
    return model, size


@torch.no_grad()
def model_probs(model, size, dataset_factory, device, batch=256, workers=0):
    """Softmax probabilities of one model over a dataset built by factory(size)."""
    loader = DataLoader(dataset_factory(size), batch_size=batch, shuffle=False,
                        num_workers=workers)
    out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x.to(device)), dim=1).cpu())
    return torch.cat(out)


def ensemble_probs(specs, dataset_factory, device):
    """Average softmax over [(model, size), ...]."""
    return torch.stack([model_probs(m, s, dataset_factory, device)
                        for m, s in specs]).mean(0)


def unnormalize(t):
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def save_grid(pairs, fname, ncols=4, title=None, cellsize=2.4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cellsize * ncols, cellsize * nrows + 0.6))
    axes = np.atleast_2d(axes)
    for i, (img, cap, correct) in enumerate(pairs):
        ax = axes[i // ncols][i % ncols]
        ax.imshow(img)
        ax.set_title(cap, fontsize=8.5,
                     color=("black" if correct is None else ("green" if correct else "red")))
        ax.axis("off")
    for j in range(len(pairs), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = os.path.join(SAMPLE_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="single checkpoint path")
    ap.add_argument("--ensemble", default=None, help="comma-separated checkpoints")
    ap.add_argument("--tag", default="best")
    ap.add_argument("--output", default=None, help="submission path (default: outputs/submission.csv)")
    ap.add_argument("--skip-analysis", action="store_true")
    ap.add_argument("--no-submit", action="store_true",
                    help="evaluate/export only, do not rewrite submission.csv")
    args = ap.parse_args()
    if not args.model and not args.ensemble:
        ap.error("provide --model or --ensemble")
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    seed_everything(0)
    device = get_device()
    paths = [args.model] if args.model else args.ensemble.split(",")
    specs = [load_model(p, device) for p in paths]

    val_factory = lambda size: datasets.ImageFolder(os.path.join(DATA, "val"),
                                                    transform=eval_tf(size))
    test_factory = lambda size: FlatImageDataset(os.path.join(DATA, "test"),
                                                 transform=eval_tf(size))

    # ---------- validation (labelled): accuracy + qualitative exports ----------
    if not args.skip_analysis:
        probs = ensemble_probs(specs, val_factory, device)
        val_ds_ref = val_factory(128)
        if val_ds_ref.class_to_idx != {"cat": 0, "dog": 1} or len(val_ds_ref) != 5000:
            raise ValueError("Unexpected teacher validation classes or image count")
        classes = val_ds_ref.classes            # ['cat', 'dog']
        ys = [y for _, y in val_ds_ref.samples]
        conf, pred = probs.max(1)
        acc = float((pred.numpy() == np.array(ys)).mean())
        print(f"[val] accuracy of this model/ensemble = {acc:.4f}")
        order = np.argsort(conf.numpy())
        correct_idx = [i for i in range(len(ys)) if pred[i] == ys[i]]
        wrong_idx = [i for i in range(len(ys)) if pred[i] != ys[i]]
        hi = sorted(correct_idx, key=lambda i: -conf[i])[:8]
        lo_conf_wrong = sorted(wrong_idx, key=lambda i: -conf[i])[:8]

        def val_img(i):
            x, _ = val_ds_ref[i]
            return unnormalize(x)

        def cap(i):
            return (f"true:{classes[ys[i]]}\n"
                    f"pred:{classes[pred[i]]} ({conf[i]:.2f})")

        save_grid([(val_img(i), cap(i), True) for i in hi],
                  f"val_correct_{args.tag}.png",
                  title="Highest-confidence correct predictions (validation set)")
        save_grid([(val_img(i), cap(i), False) for i in lo_conf_wrong],
                  f"val_wrong_{args.tag}.png",
                  title="Most confident errors (validation set)")
        np.save(os.path.join(SAMPLE_DIR, f"val_conf_{args.tag}.npy"),
                np.stack([pred.numpy(), np.array(ys), conf.numpy()], 1))

    # ---------- test set -> submission.csv ----------
    probs = ensemble_probs(specs, test_factory, device)
    conf, pred = probs.max(1)
    test_files = test_factory(specs[0][1]).files
    ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in test_files]
    if ids != list(range(1, 501)):
        raise ValueError("test IDs must be the unique integers 1..500")
    sub = pd.DataFrame({"id": ids, "label": pred.numpy()})
    sub = sub.sort_values("id").reset_index(drop=True)
    if not args.no_submit:
        out_csv = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "..", "outputs", "submission.csv")
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        sub.to_csv(out_csv, index=False)
        print(f"[saved] {out_csv} ({len(sub)} rows, predicted dog ratio="
              f"{sub.label.mean():.3f})")

    if not args.skip_analysis:
        tf128 = eval_tf(specs[0][1])
        test_ds = test_factory(specs[0][1])

        def test_img(i):
            return unnormalize(test_ds[i][0])

        order_hi = np.argsort(-conf.numpy())[:8]
        order_lo = np.argsort(conf.numpy())[:8]
        save_grid([(test_img(i),
                    f"pred:{IDX2CLS[int(pred[i])]} ({conf[i]:.2f})", None)
                   for i in order_hi], f"test_confident_{args.tag}.png",
                  title="Most confident test predictions")
        save_grid([(test_img(i),
                    f"pred:{IDX2CLS[int(pred[i])]} ({conf[i]:.2f})", None)
                   for i in order_lo], f"test_uncertain_{args.tag}.png",
                  title="Least confident test predictions")
        for rank, i in enumerate(np.argsort(conf.numpy())[:3]):
            from PIL import Image
            Image.fromarray((test_img(i) * 255).astype(np.uint8)).save(
                os.path.join(SAMPLE_DIR, f"test_uncertain_{rank}_id{ids[i]}.png"))
        np.save(os.path.join(SAMPLE_DIR, f"test_conf_{args.tag}.npy"),
                np.stack([np.array(ids), pred.numpy(), conf.numpy()], 1))


if __name__ == "__main__":
    main()
