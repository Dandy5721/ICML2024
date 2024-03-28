import re

import matplotlib.pyplot as plt
import numpy as np
import torch


def fc2vector(fc, offset=-1):
    """Vectorize the strict lower triangle of a batch of symmetric (e.g. FC) matrices."""
    index = torch.tril_indices(fc.shape[-2], fc.shape[-1], offset=offset)
    return fc[..., index[0], index[1]]


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_epochs(fname, X, epochs, xlabel, ylabel, legends, max=True):
    plt.figure()
    for i, x in enumerate(X):
        val = np.max(x) if max else np.min(x)
        idx = np.argmax(x) + 1 if max else np.argmin(x) + 1
        plt.plot(epochs, x, label=legends[i])
        plt.plot(idx, val, 'ko')
        plt.annotate(f'({idx},{val:.4f})', xy=(idx, val), xytext=(idx, val))

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(fname)
    plt.close()
