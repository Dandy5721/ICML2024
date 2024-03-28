"""ADNI disease-diagnosis, 5-fold CV, SPDNet baseline (Sec. 4.3, Table 2 middle)."""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepholobrain.data import ADataset
from deepholobrain.models import SPDNet
from scripts.common import run_kfold_classification


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fc_path', required=True, help='Dir of per-subject FC .txt files (AAL90).')
    parser.add_argument('--sc_path', required=True, help='Dir of per-subject SC .mat files.')
    parser.add_argument('--label_path', required=True, help='CSV with columns subject_id, Label1.')
    parser.add_argument('--output_path', type=str, default='adni_spdnet')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--number_cls', type=int, default=2)
    parser.add_argument('--spd', type=int, default=0, help='SPDNet depth ablation, see models.SPDNet.')
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--use_conda', type=str, default='cuda:0', help='torch device string.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.use_conda if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    dataset = ADataset(args.fc_path, args.label_path, args.sc_path)
    run_kfold_classification(
        dataset=dataset,
        model_fn=SPDNet,
        num_classes=args.number_cls,
        num_folds=args.num_folds,
        epochs=args.epochs,
        lr=args.lr,
        spd=args.spd,
        batch_size=args.batch_size,
        device=device,
        output_dir_name=args.output_path,
        shuffle_folds=False,
    )
