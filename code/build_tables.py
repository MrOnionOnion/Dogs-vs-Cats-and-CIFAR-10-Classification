"""Build report-ready comparison CSVs from saved experiment metrics."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
METRICS = OUT / 'metrics'


def read_metric(name):
    return json.loads((METRICS / (name + '.json')).read_text())


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: round(value, 6) if isinstance(value, float) else value
                          for key, value in row.items()} for row in rows)
    print(path)


def main():
    final_resnet = read_metric('dvc_resnet_final_eval')
    dvc_names = ['dvc_main_s0', 'dvc_main_s1', 'dvc_main_s2',
                 'dvc_noaug', 'dvc_small64']
    dvc_rows = []
    if all((METRICS / (name + '.json')).is_file() for name in dvc_names):
        baseline = read_metric('dvc_main_s0')['final_val_acc']
        for name in dvc_names:
            run = read_metric(name)
            dvc_rows.append(dict(run=name,
                model=run['arch'], input_px=run['size'], augmented=run['aug'],
                train_count=run['n_train'], val_count=run['n_val'],
                epochs=run['epochs'], seed=run['seed'], val_accuracy=run['final_val_acc'],
                delta_vs_main_seed0=(run['final_val_acc'] - baseline)))
        dvc_rows.append(dict(run='dvc_resnet',
            model='resnet18', input_px=160, augmented=True, train_count=10000,
            val_count=5000, epochs=8, seed=0, val_accuracy=final_resnet['validation_accuracy'],
            delta_vs_main_seed0=''))
        write_csv(OUT / 'model_processing_comparison.csv',
            ['run', 'model', 'input_px', 'augmented', 'train_count',
             'val_count', 'epochs', 'seed', 'val_accuracy', 'delta_vs_main_seed0'], dvc_rows)

    cifar_names = ['cifar10imb_baseline', 'cifar10imb_weighted', 'cifar10imb_oversamp']
    if all((METRICS / (name + '.json')).is_file() for name in cifar_names):
        rows = []
        for name in cifar_names:
            run = read_metric(name)
            rows.append(dict(method=run['method'], n_train=run['n_train'],
                n_val=run['n_val'], n_test=run['n_test'], epochs=run['epochs'],
                seed=run['seed'], best_epoch=run['best_epoch'],
                best_val_accuracy=run['best_val_acc'], test_accuracy=run['test_acc'],
                macro_recall=run['macro_recall'], minority_recall=run['minority_recall'],
                cat_recall=run['per_class_recall'][3], dog_recall=run['per_class_recall'][5],
                horse_recall=run['per_class_recall'][7],
                train_index_sha256=run['train_index_sha256'],
                val_index_sha256=run['val_index_sha256']))
        write_csv(OUT / 'imbalance_comparison.csv', list(rows[0]), rows)
    main_path = METRICS / 'cifar10_s0.json'
    if main_path.is_file():
        run = read_metric('cifar10_s0')
        rows = [dict(class_name=name, support=sum(run['confusion_matrix'][i]),
                     correct=run['confusion_matrix'][i][i], recall=run['per_class_acc'][i])
                for i, name in enumerate(run['classes'])]
        write_csv(OUT / 'cifar_main_per_class.csv', list(rows[0]), rows)


if __name__ == '__main__':
    main()
