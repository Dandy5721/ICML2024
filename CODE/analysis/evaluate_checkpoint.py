"""Evaluate one saved fold checkpoint (any dataset) and persist per-class logits.

Reproduces the exact test split `run_kfold_classification` used for
`--fold`, so `--shuffle_folds` must match the value used to produce the
checkpoint: HCP-A, OASIS... train with the default (shuffled) folds,
`train_diagnosis_adni.py` uses `--shuffle_folds 0`.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepholobrain.data import ADataset, FC_SCDataset, FCSCDataset, OASISDataset
from deepholobrain.models import DeepHoloBrain, SPDNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=('hcpa', 'adni', 'oasis'), required=True)
    p.add_argument('--model', choices=('spdnet', 'deepholobrain'), required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--fc_path', required=True)
    p.add_argument('--sc_path', required=True)
    p.add_argument('--label_path', help='Required for --dataset adni/oasis.')
    p.add_argument('--num_classes', type=int, required=True)
    p.add_argument('--num_folds', type=int, default=5)
    p.add_argument('--fold', type=int, required=True)
    p.add_argument('--shuffle_folds', type=int, choices=(0, 1), default=1,
                    help='Must match the KFold shuffle setting used to produce the checkpoint.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--spd', type=int, default=1)
    p.add_argument('--manifold_dim', type=int, choices=(16, 32, 64), default=16)
    p.add_argument('--model_variant', choices=('optimized', 'deepholobrain'), default='optimized')
    p.add_argument('--enable_mlp_mixer', action='store_true')
    p.add_argument('--enable_location_frequency_attention', action='store_true')
    p.add_argument('--gamma_init', type=float, default=1.0)
    p.add_argument('--num_frequencies', type=int)
    p.add_argument('--gated_scattering', action='store_true')
    p.add_argument('--classifier_hidden_dim', type=int, default=0)
    p.add_argument('--classifier_dropout', type=float, default=0.0)
    p.add_argument('--output', required=True)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--device', default='cuda:0')
    return p.parse_args()


def build_dataset(args):
    if args.dataset == 'hcpa':
        if args.model == 'spdnet':
            return FCSCDataset(args.fc_path)
        return FC_SCDataset(
            args.fc_path, args.sc_path,
            sc_normalization='total' if args.model_variant == 'deepholobrain' else 'row',
        )
    if not args.label_path:
        raise ValueError('--label_path is required for --dataset adni/oasis')
    if args.dataset == 'adni':
        return ADataset(args.fc_path, args.label_path, args.sc_path)
    return OASISDataset(args.fc_path, args.label_path, args.sc_path)


def build_model(args):
    if args.model == 'spdnet':
        return SPDNet(args.num_classes)
    deepholobrain = args.model_variant == 'deepholobrain'
    return DeepHoloBrain(
        args.num_classes, gated_scattering=args.gated_scattering,
        manifold_dim=args.manifold_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        residual_fc=not deepholobrain,
        deepholobrain=deepholobrain,
        mlp_mixer=args.enable_mlp_mixer,
        gamma_init=args.gamma_init,
        location_frequency_attention=args.enable_location_frequency_attention,
        num_frequencies=args.num_frequencies,
        classifier_dropout=args.classifier_dropout,
    )


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dataset = build_dataset(args)
    model = build_model(args)

    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if 'model' in state:
        state = state['model']
    model.load_state_dict(state)
    model.to(device).eval()

    shuffle_folds = bool(args.shuffle_folds)
    split = list(KFold(
        args.num_folds, shuffle=shuffle_folds, random_state=args.seed if shuffle_folds else None,
    ).split(dataset))[args.fold - 1]
    test_idx = split[1]
    loader = DataLoader(
        Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    targets, logits = [], []
    with torch.no_grad():
        for batch in loader:
            if args.model == 'spdnet':
                x, y = batch[0], batch[1]  # datasets yield (x, y) or (x, y, sc)
                _, out = model(x.to(device), args.spd)
            else:
                x, y, sc = batch
                _, out, _, _ = model(x.to(device), sc.to(device), args.spd)
            targets.extend(y.numpy())
            logits.extend(out.cpu().numpy())

    targets = np.asarray(targets)
    logits = np.asarray(logits)
    prediction = logits.argmax(1)
    frame = pd.DataFrame({
        'dataset_index': test_idx, 'target': targets, 'prediction': prediction,
        **{f'logit_{i}': logits[:, i] for i in range(logits.shape[1])},
    })
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    frame.to_csv(args.output, index=False)
    print('accuracy', accuracy_score(targets, prediction))
    print('macro_recall', recall_score(targets, prediction, average='macro'))
    print('macro_f1', f1_score(targets, prediction, average='macro'))
    print('saved', args.output)


if __name__ == '__main__':
    main()
