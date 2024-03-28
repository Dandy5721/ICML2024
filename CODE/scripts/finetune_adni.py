"""Fine-tune a HCP-A-pretrained SPDNet layer stack for ADNI diagnosis
(Sec. 4.3, "Generality as a Pre-trained Model", Table 2 bottom / 'baseline+').

Loads a checkpoint produced by `pretrain_moca_hcpa_spdnet.py` (or any script
saving a `SPDNet`/`DeepHoloBrain` state_dict), lifts out the sub-module whose
attribute name matches `--layer_name2` (e.g. `layers1`, the depth-1 SPDNet
stack), and trains a small classifier on top of it.
"""

import argparse
import os
import sys

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepholobrain.data import ADataset
from deepholobrain.layers import Normalize, SPDRectified, SPDTangentSpace, SPDTransform
from deepholobrain.optim import StiefelMetaOptimizer


class FinetuneModel(nn.Module):
    """SPDTransform(116, 64) -> SPDRectified -> SPDTangentSpace -> Normalize -> Linear."""

    def __init__(self, num_classes, pretrained_state_dict):
        super(FinetuneModel, self).__init__()
        self.layers1 = nn.Sequential(
            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTangentSpace(vectorize_all=False),
            Normalize(),
        )
        self.layers1.load_state_dict(pretrained_state_dict)
        self.classifier1 = nn.Linear(64 * 65 // 2, num_classes)

    def forward(self, x):
        x = self.layers1(x)
        x = x.view(x.size(0), -1)
        return self.classifier1(x)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fc_path', required=True, help='Dir of per-subject FC .txt files (AAL116).')
    parser.add_argument('--sc_path', required=True, help='Dir of per-subject SC .mat files.')
    parser.add_argument('--label_path', required=True, help='CSV with columns subject_id, Label1.')
    parser.add_argument('--model_path', required=True, help='Pretrained checkpoint (as saved by pretrain_moca_hcpa_spdnet.py).')
    parser.add_argument('--layer_name', type=str, default='layers1.', help="Prefix (with trailing dot) to strip from matching state_dict keys.")
    parser.add_argument('--layer_name2', type=str, default='layers1', help='Substring identifying which sub-module to transplant.')
    parser.add_argument('--out_path', required=True, help='Where to write the best-epoch metrics CSV.')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--number_cls', type=int, default=2)
    parser.add_argument('--use_conda', type=str, default='cuda:0', help='torch device string.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.use_conda if torch.cuda.is_available() else 'cpu')

    pretrained_state_dict = torch.load(args.model_path)['model']
    transplant_state_dict = {
        name.replace(args.layer_name, ''): param
        for name, param in pretrained_state_dict.items()
        if args.layer_name2 in name
    }

    num_sample = len([p for p in os.listdir(args.fc_path) if not p.startswith('.')])
    num_train = num_sample // 2
    num_test = num_sample - num_train
    train_dataset = ADataset(args.fc_path, args.label_path, args.sc_path, slice=slice(num_train))
    test_dataset = ADataset(args.fc_path, args.label_path, args.sc_path, slice=slice(-num_test, None))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

    model = FinetuneModel(args.number_cls, transplant_state_dict).to(device)
    optimizer = StiefelMetaOptimizer(SGD(model.parameters(), lr=args.lr, weight_decay=1e-5, momentum=0.9))
    criterion = nn.CrossEntropyLoss()

    best_metrics = {'best_test_acc': 0, 'best_test_recall': 0, 'best_test_f1': 0, 'best_test_precision': 0, 'best_test_auc': 0}

    for epoch in range(args.epochs):
        model.train()
        for inputs, labels, _sc in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}'):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        test_labels, test_preds, test_probs = [], [], []
        with torch.no_grad():
            for inputs, labels, _sc in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[:, 1]
                test_probs.extend(probs)
                test_preds.extend(outputs.argmax(-1).cpu().numpy())
                test_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(test_labels, test_preds)
        test_recall = recall_score(test_labels, test_preds, average='weighted', zero_division=1)
        test_f1 = f1_score(test_labels, test_preds, average='weighted', zero_division=1)
        test_precision = precision_score(test_labels, test_preds, average='weighted', zero_division=1)
        test_auc = roc_auc_score(test_labels, test_probs)
        print(
            f'Epoch {epoch + 1}/{args.epochs} - Test Acc: {test_acc:.4f}, Recall: {test_recall:.4f}, '
            f'F1: {test_f1:.4f}, Precision: {test_precision:.4f}, AUC: {test_auc:.4f}'
        )

        if test_acc > best_metrics['best_test_acc']:
            best_metrics = {
                'best_test_acc': test_acc,
                'best_test_recall': test_recall,
                'best_test_f1': test_f1,
                'best_test_precision': test_precision,
                'best_test_auc': test_auc,
            }

    pd.DataFrame([best_metrics]).to_csv(args.out_path, index=False)
    print('Best metrics saved to', args.out_path)
