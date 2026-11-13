# EE6483 Mini Project Option 2 — Experiment Materials

This repository contains the Dogs vs. Cats and CIFAR-10 experiment code, model checkpoints, training logs, figures, metrics, and the final `submission.csv`. See [`实验数据与结果.md`](实验数据与结果.md) for the configurations and result tables. The coursework specification was supplied separately by the instructor.

## Datasets

- **Dogs vs. Cats:** Use the teacher-provided [`datasets.zip` from the coursework Google Drive link](https://drive.google.com/file/d/1q0r6yeHQMS17R3wz-s2FIbMR5DAGZK5v/view?usp=sharing), not the original Kaggle archive. The archive is approximately 1.1 GB and is not duplicated in this repository. SHA-256: `1b5295e37aac84c05fd59cf471282c76dc7c17b3ac9d81f1df0bd232d2158582`. Extract its `datasets/` directory into the repository root, or set `EE6483_DATA_DIR` to that directory.
- **CIFAR-10:** The [official University of Toronto dataset](https://www.cs.toronto.edu/~kriz/cifar.html) is downloaded and checked by `python code/download_assets.py`.

## Included files

- `code/`: training, model definitions, inference, asset download, and table generation.
- `ckpts/dvc_resnet.pt`: selected Dogs vs. Cats ResNet-18 model.
- `outputs/ckpts/`, `outputs/metrics/`, `outputs/logs/`, `outputs/figures/`: comparison models and experimental results.
- `submission.csv`: 500 Dogs vs. Cats Test predictions, with `cat=0` and `dog=1`.

The teacher's Dogs vs. Cats Test images do not have public labels. The reported **98.22%** accuracy is on the labeled 5,000-image Validation set.

## Run

Use Python 3.10 or 3.11 with compatible PyTorch and torchvision, NumPy, Pillow, pandas, and matplotlib. After extracting the teacher dataset, examples are:

```bash
python code/train_dvc.py --mode main --seed 0
python code/train_cifar.py --seed 0 --epochs 30
python code/train_cifar_imb.py --method weighted --seed 0 --epochs 16
python code/predict_test.py --model ckpts/dvc_resnet.pt --output submission.csv --skip-analysis
```

The repository contains the code and saved results; it does not include the teacher image archive or the downloaded CIFAR-10 cache.
