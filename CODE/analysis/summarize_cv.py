"""Recompute macro-averaged cross-validation metrics from saved fold predictions.

Works on any `run_kfold_classification` output directory (HCP-A, ADNI or
OASIS): prediction-driven rather than trusting a saved metrics file.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--num-folds', type=int, default=5)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--fold-dir-template',
                        help='Optional directory template containing {fold}, for parallel per-fold runs.')
    parser.add_argument('--output', type=Path,
                        help='Output CSV path (default: result_dir/cross_validation_macro_results.csv).')
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    completed_folds = []
    for fold in range(1, args.num_folds + 1):
        fold_dir = Path(args.fold_dir_template.format(fold=fold)) if args.fold_dir_template else args.result_dir
        path = fold_dir / f'fold_{fold}_predictions.csv'
        if not path.exists():
            if args.allow_partial:
                continue
            raise FileNotFoundError(f'Missing fold prediction file: {path}')
        predictions = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.float64)
        predictions = np.atleast_2d(predictions)
        target = predictions[:, 1].astype(np.int64)
        predicted = predictions[:, 2].astype(np.int64)
        rows.append([
            accuracy_score(target, predicted),
            recall_score(target, predicted, average='macro', zero_division=0),
            f1_score(target, predicted, average='macro', zero_division=0),
        ])
        completed_folds.append(fold)

    if not rows:
        raise RuntimeError(f'No fold prediction files found under {args.result_dir}')

    fold_metrics = np.asarray(rows)
    summary = np.vstack([fold_metrics, fold_metrics.mean(axis=0), fold_metrics.std(axis=0)])
    output = args.output or (args.result_dir / 'cross_validation_macro_results.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output, summary, fmt='%.8f', delimiter=',',
        header='Accuracy,MacroRecall,MacroF1', comments='',
    )
    print('completed_folds:', completed_folds)
    print('per_fold:\n', fold_metrics)
    print('mean:', fold_metrics.mean(axis=0))
    print('std:', fold_metrics.std(axis=0))
    print('saved:', output)


if __name__ == '__main__':
    main()
