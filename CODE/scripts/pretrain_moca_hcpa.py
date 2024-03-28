"""Pretrain DeepHoloBrain to regress MoCA score on HCP-A (Sec. 4.3, "Generality
as a Pre-trained Model"). The resulting checkpoint is fine-tuned on ADNI by
`scripts/finetune_adni.py`.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepholobrain.data import HCPADataset_pre
from deepholobrain.models import DeepHoloBrain
from scripts.common import run_holdout


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fc_path', required=True, help='Dir of per-subject FC .csv files (AAL116).')
    parser.add_argument('--sc_path', required=True, help='Dir of per-subject SC .mat folders.')
    parser.add_argument('--label_path', required=True, help='CSV with columns src_subject_id, moca_total.')
    parser.add_argument('--output_path', type=str, default='hcpa_deepholobrain_moca')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--spd', type=int, default=1, help='1: manifold layers + classifier1, 0: MLP classifier2.')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--use_conda', type=str, default='cuda:0', help='torch device string.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.use_conda if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    dataset = HCPADataset_pre(args.fc_path, args.label_path, args.sc_path)
    num_sample = len(dataset)
    num_train = num_sample // 10 * 7
    num_val = (num_sample - num_train) // 2
    num_test = num_sample - num_train - num_val
    print(f'num_sample={num_sample} num_train={num_train} num_val={num_val} num_test={num_test}')

    train_dataset = HCPADataset_pre(args.fc_path, args.label_path, args.sc_path, slice=slice(num_train))
    val_dataset = HCPADataset_pre(
        args.fc_path, args.label_path, args.sc_path, slice=slice(num_train, num_train + num_val)
    )
    test_dataset = HCPADataset_pre(args.fc_path, args.label_path, args.sc_path, slice=slice(-num_test, None))

    model = DeepHoloBrain(num_classes=1)
    run_holdout(
        train_dataset, val_dataset, test_dataset, model, task='regression',
        epochs=args.epochs, lr=args.lr, spd=args.spd, device=device,
        output_dir_name=args.output_path, batch_size=args.batch_size,
    )
