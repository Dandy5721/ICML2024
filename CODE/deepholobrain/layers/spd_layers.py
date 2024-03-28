"""Building blocks for SPD-manifold networks (Sec. 2.2, cf. SPDNet).

`SPDTransform` implements the positive mapping f(X, W) = W^T X W of Eq. 1
(Huang & Van Gool, 2017), which every DeepHoloBrain / SPDNet layer stack is
built from. `SPDTangentSpace` is the Riemannian log map at the identity,
projecting an SPD matrix to its tangent space for a standard classifier.
"""

import numpy as np
import torch
from torch import nn

from ..optim.manifold_parameters import StiefelParameter
from .riemannian import frechet_mean, safe_eigh, sqrtm


class SPDTransform(nn.Module):
    """Positive mapping f(X, W) = W^T X W (Eq. 1). W is constrained to the Stiefel manifold."""

    def __init__(self, input_size, output_size, in_channels=1):
        super(SPDTransform, self).__init__()

        if in_channels > 1:
            self.weight = StiefelParameter(
                torch.Tensor(in_channels, input_size, output_size), requires_grad=True
            )
        else:
            self.weight = StiefelParameter(
                torch.Tensor(input_size, output_size), requires_grad=True
            )
        nn.init.orthogonal_(self.weight)

    def forward(self, input):
        weight = self.weight
        return weight.transpose(-2, -1) @ input @ weight


class SPDVectorize(nn.Module):
    """Vectorize the upper triangle (incl. diagonal) of a batch of symmetric matrices."""

    def __init__(self, vectorize_all=True):
        super(SPDVectorize, self).__init__()
        self.register_buffer('vectorize_all', torch.tensor(vectorize_all))

    def forward(self, input):
        row_idx, col_idx = np.triu_indices(input.shape[-1])
        output = input[..., row_idx, col_idx]
        if self.vectorize_all:
            output = torch.flatten(output, 1)
        return output


class SPDTangentSpace(nn.Module):
    """Riemannian log map at the identity: project an SPD matrix to its tangent space."""

    def __init__(self, vectorize=True, vectorize_all=True, eigh_eps=1e-4):
        super(SPDTangentSpace, self).__init__()
        self.vectorize = vectorize
        self.eigh_eps = eigh_eps
        if vectorize:
            self.vec = SPDVectorize(vectorize_all=vectorize_all)

    def forward(self, input):
        s, u = safe_eigh(input, self.eigh_eps)
        s = s.log().diag_embed()
        if s.isnan().any():
            raise ValueError('SPDTangentSpace: log of a non-positive eigenvalue produced NaN')
        output = u @ s @ u.transpose(-2, -1)

        if self.vectorize:
            output = self.vec(output)
        return output


class SPDRectified(nn.Module):
    """Clamp eigenvalues away from zero to keep a matrix strictly positive-definite."""

    def __init__(self, epsilon=1e-4):
        super(SPDRectified, self).__init__()
        self.register_buffer('epsilon', torch.DoubleTensor([epsilon]))

    def forward(self, input):
        s, u = safe_eigh(input, self.epsilon[0].item())
        s = s.clamp(min=self.epsilon[0]).diag_embed()
        return u @ s @ u.transpose(-2, -1)


class Normalize(nn.Module):
    def __init__(self, p=2, dim=-1):
        super(Normalize, self).__init__()
        self.p = p
        self.dim = dim

    def forward(self, input):
        norm = input.norm(self.p, self.dim, keepdim=True)
        return input / norm


class SPDNormalization(nn.Module):
    """Recenter a batch of SPD matrices at their Frechet mean (Riemannian batch-norm)."""

    def __init__(self, input_size):
        super(SPDNormalization, self).__init__()

    def forward(self, input):
        center = frechet_mean(input, num_iter=1)
        center_sqrt_inv = sqrtm(center).inverse()
        return center_sqrt_inv @ input @ center_sqrt_inv
