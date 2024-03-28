"""Riemannian optimizer wrapper for parameters living on the Stiefel or SPD manifold.

Wraps any Euclidean `torch.optim.Optimizer` (e.g. SGD): Euclidean gradients on
manifold-constrained parameters (`StiefelParameter`, `SPDParameter`) are
projected onto the tangent space before the base optimizer's step, and the
updated parameter is retracted (Stiefel) / exponential-mapped (SPD) back onto
the manifold afterwards.
"""

import torch

from ..layers.riemannian import expm
from .manifold_parameters import SPDParameter, StiefelParameter


def orthogonal_projection(A, B):
    """Project A onto the tangent space of the Stiefel manifold at B."""
    return A - B @ A.transpose(-2, -1) @ B


def retraction(A, ref=None):
    """Retract a tangent vector back onto the Stiefel manifold via QR."""
    data = A if ref is None else A + ref
    Q, R = torch.linalg.qr(data, mode='reduced')
    sign = (R.diagonal(dim1=-2, dim2=-1).sign() + 0.5).sign().diag_embed()
    return Q @ sign


class StiefelMetaOptimizer:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.state = {}

    def zero_grad(self):
        return self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Restore the wrapped torch optimizer for resumable experiments."""
        return self.optimizer.load_state_dict(state_dict)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.grad[torch.isnan(p.grad)] = 0.0
                if isinstance(p, StiefelParameter):
                    trans = orthogonal_projection(p.grad, p)
                    p.grad.fill_(0).add_(trans)
                elif isinstance(p, SPDParameter):
                    riem = p @ ((p.grad + p.grad.transpose(-2, -1)) / 2) @ p
                    self.state[p] = p.clone()
                    p.fill_(0)
                    p.grad.fill_(0).add_(riem)

        loss = self.optimizer.step(closure)

        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if isinstance(p, StiefelParameter):
                    trans = retraction(p)
                    p.fill_(0).add_(trans)
                elif isinstance(p, SPDParameter):
                    trans = expm(self.state[p], p)
                    p.fill_(0).add_(trans)

        return loss
