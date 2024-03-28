"""Shared training/evaluation loops for the entry-point scripts in this folder.

`run_kfold_classification` is the k-fold CV loop (used by all `train_*`
scripts); `run_holdout` is the single train/val/test-split loop (used by
the `pretrain_*` regression scripts).
"""

import fcntl
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from deepholobrain.optim import StiefelMetaOptimizer
from deepholobrain.utils import AverageMeter, plot_epochs

LOG_COLUMNS = ['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'val_loss', 'val_acc']


class FocalCrossEntropy(nn.Module):
    """Multiclass focal loss with optional static class weights."""

    def __init__(self, gamma, weight=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer('weight', weight)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction='none')
        probability = F.softmax(logits, dim=-1).gather(1, target[:, None]).squeeze(1)
        return ((1.0 - probability).pow(self.gamma) * ce).mean()


def _unpack_batch(batch):
    """Datasets yield either (x, y) or (x, y, sc); SPDNet only ever needs (x, y)."""
    if len(batch) == 3:
        inputs, targets, _sc = batch
        return inputs, targets
    return batch


def make_result_dirs(output_dir_name, window=None):
    suffix = f'window={window}' if window is not None else ''
    train_result_path = os.path.join('train_results', output_dir_name, suffix)
    test_result_path = os.path.join('test_results', output_dir_name, suffix)
    os.makedirs(test_result_path, exist_ok=True)
    os.makedirs(os.path.join(train_result_path, 'models_save'), exist_ok=True)
    return train_result_path, test_result_path


def run_kfold_classification(
    dataset, model_fn, num_classes, num_folds, epochs, lr, spd, batch_size,
    device, output_dir_name, num_workers=8, shuffle_folds=True, shuffle_train=True, seed=42,
    scale_regularization_beta=1.0, only_fold=None, class_weighting='none', lr_scheduler='none',
    resume=False, class_weight_power=None, focal_gamma=0.0, label_smoothing=0.0,
    optimizer_name='sgd', weight_decay=1e-5,
):
    """K-fold CV classification loop (Sec. 4.2/4.3), shared by the SPDNet and
    DeepHoloBrain `train_*` scripts. `only_fold` runs a single fold (for
    parallel evaluation); `resume` continues from the latest checkpoint.
    """
    test_result_path = os.path.join('test_results', output_dir_name)
    os.makedirs(test_result_path, exist_ok=True)
    model_result_path = os.path.join('train_results', output_dir_name, 'models_save')
    os.makedirs(model_result_path, exist_ok=True)

    kf = KFold(n_splits=num_folds, shuffle=shuffle_folds, random_state=seed if shuffle_folds else None)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset), start=1):
        if only_fold is not None and fold != only_fold:
            continue
        lock_path = os.path.join(model_result_path, f'fold_{fold}.lock')
        lock_handle = open(lock_path, 'w')
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_handle.close()
            raise RuntimeError(f'Fold {fold} for output {output_dir_name!r} is already running') from exc
        print(f'Starting fold {fold}')

        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers,
        )
        test_loader = DataLoader(
            Subset(dataset, test_idx), batch_size=batch_size, shuffle=False, num_workers=num_workers,
        )

        model = model_fn(num_classes).to(device)
        is_holobrain = model.__class__.__name__ == 'DeepHoloBrain'
        if optimizer_name == 'sgd':
            base_optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_name == 'adamw':
            base_optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f'Unsupported optimizer_name={optimizer_name!r}')
        optimizer = StiefelMetaOptimizer(base_optimizer)
        if lr_scheduler == 'cosine':
            scheduler = CosineAnnealingLR(base_optimizer, T_max=epochs, eta_min=lr * 0.01)
        elif lr_scheduler == 'none':
            scheduler = None
        else:
            raise ValueError(f'Unsupported lr_scheduler={lr_scheduler!r}')
        if class_weighting in ('balanced', 'sqrt_balanced') or class_weight_power is not None:
            labels = np.asarray(dataset.labels, dtype=np.int64)[train_idx]
            counts = np.bincount(labels, minlength=num_classes)
            if np.any(counts == 0):
                raise ValueError(f'Fold {fold} has an empty class: counts={counts.tolist()}')
            weights = len(train_idx) / (num_classes * counts)
            if class_weight_power is not None:
                if not 0.0 <= class_weight_power <= 1.0:
                    raise ValueError('class_weight_power must be between 0 and 1')
                weights = weights ** class_weight_power
                weighting_name = f'class_weight_power={class_weight_power:g}'
            elif class_weighting == 'sqrt_balanced':
                weights = np.sqrt(weights)
                weighting_name = class_weighting
            else:
                weighting_name = class_weighting
            weight_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(
                weight=weight_tensor, label_smoothing=label_smoothing,
            ) if focal_gamma <= 0 else FocalCrossEntropy(focal_gamma, weight_tensor)
            print(f'Fold {fold} {weighting_name} class weights: {weights.tolist()}')
        elif class_weighting == 'none':
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing) \
                if focal_gamma <= 0 else FocalCrossEntropy(focal_gamma)
        else:
            raise ValueError(f'Unsupported class_weighting={class_weighting!r}')

        latest_path = os.path.join(model_result_path, f'fold_{fold}_latest.pt')
        history_path = os.path.join(test_result_path, f'fold_{fold}_training_history.csv')
        start_epoch = 0
        training_history = []
        if resume and os.path.exists(latest_path):
            checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = int(checkpoint['epoch'])
            if scheduler is not None:
                if 'scheduler' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                else:
                    for _ in range(start_epoch):
                        scheduler.step()
            if os.path.exists(history_path):
                history = np.loadtxt(history_path, delimiter=',', skiprows=1, ndmin=2)
                training_history = history[:start_epoch].tolist()
            print(f'Resuming fold {fold} from completed epoch {start_epoch}')

        for epoch in range(start_epoch, epochs):
            model.train()
            losses = AverageMeter()
            epoch_targets, epoch_predictions = [], []
            bar = tqdm(train_loader, desc=f'fold {fold} epoch {epoch}')
            for batch in bar:
                if is_holobrain:
                    inputs, targets, sc = batch
                    sc = sc.to(device)
                else:
                    inputs, targets = _unpack_batch(batch)
                    sc = None
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                if is_holobrain:
                    scale, outputs, _, _ = model(inputs, sc, spd)
                    # Eq. 4 penalizes gamma only when it is negative.
                    regularizer = scale_regularization_beta * torch.relu(-scale).sum()
                else:
                    _, outputs = model(inputs, spd)
                    regularizer = torch.zeros((), device=device)
                loss = criterion(outputs, targets) + regularizer
                loss.backward()
                optimizer.step()

                pred = outputs.argmax(-1).cpu().numpy()
                acc = accuracy_score(targets.cpu().numpy(), pred)
                epoch_targets.extend(targets.cpu().numpy())
                epoch_predictions.extend(pred)
                losses.update(loss.item())
                bar.set_postfix(loss=f'{losses.avg:.4f}', acc=f'{acc:.4f}')

            training_history.append([
                epoch + 1, losses.avg, accuracy_score(epoch_targets, epoch_predictions),
            ])
            np.savetxt(
                history_path, np.asarray(training_history), fmt=['%d', '%.8f', '%.8f'], delimiter=',',
                header='epoch,train_loss,train_accuracy', comments='',
            )
            if scheduler is not None:
                scheduler.step()
            temporary_path = latest_path + '.tmp'
            checkpoint = {'epoch': epoch + 1, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            if scheduler is not None:
                checkpoint['scheduler'] = scheduler.state_dict()
            torch.save(checkpoint, temporary_path)
            os.replace(temporary_path, latest_path)

        torch.save(model.state_dict(), os.path.join(model_result_path, f'fold_{fold}_final.pt'))

        model.eval()
        all_targets, all_predictions, all_logits = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                if is_holobrain:
                    inputs, targets, sc = batch
                    sc = sc.to(device)
                else:
                    inputs, targets = _unpack_batch(batch)
                    sc = None
                inputs, targets = inputs.to(device), targets.to(device)
                if is_holobrain:
                    _, outputs, _, _ = model(inputs, sc, spd)
                else:
                    _, outputs = model(inputs, spd)
                predicted = outputs.argmax(-1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_logits.extend(outputs.cpu().numpy())

        acc = accuracy_score(all_targets, all_predictions)
        # Table 1 uses macro averaging (weighted recall is mathematically
        # identical to accuracy for single-label multiclass classification).
        recall = recall_score(all_targets, all_predictions, average='macro')
        f1 = f1_score(all_targets, all_predictions, average='macro')
        print(f'Fold {fold} - Accuracy: {acc:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}')
        np.savetxt(
            os.path.join(test_result_path, f'fold_{fold}_predictions.csv'),
            np.column_stack([test_idx, all_targets, all_predictions, np.asarray(all_logits)]),
            fmt=['%d', '%d', '%d'] + ['%.8g'] * num_classes, delimiter=',',
            header='dataset_index,target,prediction,' + ','.join(f'logit_{label}' for label in range(num_classes)),
            comments='',
        )
        np.savetxt(
            os.path.join(test_result_path, f'fold_{fold}_confusion_matrix.csv'),
            confusion_matrix(all_targets, all_predictions, labels=range(num_classes)),
            fmt='%d', delimiter=',',
        )
        fold_results.append([acc, recall, f1])
        np.savetxt(
            os.path.join(test_result_path, f'fold_{fold}_metrics.csv'),
            np.array([[acc, recall, f1]]), fmt='%f', delimiter=',', header='Accuracy,Recall,F1', comments='',
        )

    fold_results = np.array(fold_results)
    if only_fold is None:
        all_results = np.vstack([fold_results, fold_results.mean(axis=0), fold_results.std(axis=0)])
        np.savetxt(
            os.path.join(test_result_path, 'cross_validation_results.csv'),
            all_results, fmt='%f', delimiter=',', header='Accuracy,Recall,F1', comments='',
        )
    print('Cross-validation results saved to', test_result_path)
    return fold_results


def run_holdout(
    train_dataset, val_dataset, test_dataset, model, task, epochs, lr, spd,
    device, output_dir_name, window=30, batch_size=16, num_workers=8,
):
    """Single train/val/test-split loop (Sec. 4.3, MoCA pretraining).

    `task` is 'classification' (CrossEntropyLoss) or 'regression' (MSELoss).
    `model` may be `DeepHoloBrain` (needs `sc`, returns attention maps) or
    an `SPDNet` baseline.
    """
    assert task in ('classification', 'regression')
    is_holobrain = model.__class__.__name__ == 'DeepHoloBrain'

    train_result_path, test_result_path = make_result_dirs(output_dir_name, window)
    models_save_path = os.path.join(train_result_path, 'models_save')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    model = model.to(device)
    optimizer = StiefelMetaOptimizer(SGD(model.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9))
    criterion = nn.CrossEntropyLoss() if task == 'classification' else nn.MSELoss()

    pd.DataFrame(columns=LOG_COLUMNS).to_csv(os.path.join(train_result_path, 'log.csv'), index=False)

    def forward(inputs, sc):
        if is_holobrain:
            feature, outputs, row, column = model(inputs, sc, spd)
            regularizer = torch.relu(-feature).sum()  # Eq. 4
            return outputs, regularizer, row, column
        feature, outputs = model(inputs, spd)
        return outputs, torch.zeros((), device=inputs.device), None, None

    def metrics(y_true, y_pred, average):
        if task == 'classification':
            acc = accuracy_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred, average=average)
            f1 = f1_score(y_true, y_pred, average=average)
        else:
            acc = mean_absolute_error(y_true, y_pred)
            recall = r2_score(y_true, y_pred)
            f1 = mean_squared_error(y_true, y_pred)
        return acc, recall, f1

    def run_epoch(data_loader, train):
        model.train(train)
        losses = AverageMeter()
        batch_metrics = AverageMeter() if train else None
        all_targets, all_predictions = [], []
        row = column = None
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            bar = tqdm(data_loader, desc='train' if train else 'eval')
            for inputs, targets, sc in bar:
                inputs = inputs.squeeze().to(device)
                targets = (targets.squeeze() if task == 'regression' else targets).to(device)
                sc = sc.squeeze().to(device)

                if train:
                    optimizer.zero_grad()
                outputs, regularizer, row, column = forward(inputs, sc)
                loss = criterion(outputs, targets) + regularizer
                if train:
                    loss.backward()
                    optimizer.step()

                pred = outputs.argmax(-1) if task == 'classification' else outputs
                targets_np = targets.detach().cpu().numpy().flatten()
                pred_np = pred.detach().cpu().numpy().flatten()
                losses.update(loss.item())

                if train:
                    batch_acc, _, _ = metrics(targets_np, pred_np, average='macro')
                    batch_metrics.update(batch_acc)
                    bar.set_postfix(loss=f'{losses.avg:.4f}', acc=f'{batch_metrics.avg:.4f}')
                else:
                    all_targets.extend(targets_np)
                    all_predictions.extend(pred_np)
                    bar.set_postfix(loss=f'{losses.avg:.4f}')

        if train:
            return losses.avg, batch_metrics.avg, row, column
        acc, recall, f1 = metrics(all_targets, all_predictions, average='weighted')
        return losses.avg, acc, recall, f1, row, column

    history = {k: [] for k in ('epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'val_loss', 'val_acc')}
    best_metric = 0 if task == 'classification' else float('inf')
    is_better = (lambda new, best: new > best) if task == 'classification' else (lambda new, best: new < best)

    for epoch in range(1, epochs + 1):
        print(f'\nEpoch: {epoch}')
        train_loss, train_acc, _, _ = run_epoch(train_loader, train=True)
        test_loss, test_acc, test_recall, test_f1, row, column = run_epoch(test_loader, train=False)
        val_loss, val_acc, _, _, _, _ = run_epoch(val_loader, train=False)

        if is_better(val_acc, best_metric):
            best_metric = val_acc
            np.savetxt(
                os.path.join(test_result_path, 'accs.csv'),
                np.array([[test_acc, test_recall, test_f1]]), fmt='%f', delimiter=',',
                header='Accuracy,Recall,F1-Score', comments='',
            )
            if is_holobrain:
                np.savetxt(os.path.join(test_result_path, 'attention_row.csv'), row.cpu().numpy(), fmt='%f', delimiter=',')
                np.savetxt(os.path.join(test_result_path, 'attention_column.csv'), column.cpu().numpy(), fmt='%f', delimiter=',')

        for key, value in zip(
            history.keys(), [epoch, train_loss, train_acc, test_loss, test_acc, val_loss, val_acc]
        ):
            history[key].append(value)

        plot_epochs(
            os.path.join(train_result_path, 'loss.svg'),
            [history['train_loss'], history['test_loss'], history['val_loss']],
            history['epoch'], xlabel='epoch', ylabel='loss', legends=['train', 'test', 'val'], max=False,
        )
        plot_epochs(
            os.path.join(train_result_path, 'acc.svg'),
            [history['train_acc'], history['test_acc'], history['val_acc']],
            history['epoch'], xlabel='epoch', ylabel='accuracy', legends=['train', 'test', 'val'],
        )
        pd.DataFrame([[history[k][-1] for k in LOG_COLUMNS]]).to_csv(
            os.path.join(train_result_path, 'log.csv'), mode='a', index=False, header=False,
        )
        torch.save(
            {'epoch': epoch, 'model': model.state_dict(), 'opt': optimizer.state_dict()},
            os.path.join(models_save_path, f'{epoch}_{test_acc:.4f}.pt'),
        )

    return history
