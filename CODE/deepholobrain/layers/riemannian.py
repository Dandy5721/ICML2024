"""Elementary operations on the Riemannian manifold of SPD matrices (Sym+_d).

Matrix log/exp, the tangent/exponential maps between two points, and the
Frechet mean under the affine-invariant metric, used by the Stiefel/SPD
optimizer (`deepholobrain.optim`) and `SPDNormalization`.
"""

import torch


class _SafeEighFunction(torch.autograd.Function):
    """`torch.linalg.eigh` with a Lorentzian-smoothed backward, stable when
    eigenvalues are close or tied (`1/diff` replaced by `diff/(diff^2+eps^2)`)."""

    @staticmethod
    def forward(ctx, X, eps):
        eigvals, eigvecs = torch.linalg.eigh(X)
        ctx.save_for_backward(eigvals, eigvecs)
        ctx.eps = eps
        return eigvals, eigvecs

    @staticmethod
    def backward(ctx, grad_eigvals, grad_eigvecs):
        eigvals, eigvecs = ctx.saved_tensors
        eps = ctx.eps
        n = eigvals.shape[-1]

        diff = eigvals.unsqueeze(-2) - eigvals.unsqueeze(-1)  # diff[..., i, j] = lambda_j - lambda_i
        F = diff / (diff * diff + eps * eps)
        diag_idx = torch.arange(n, device=eigvals.device)
        F[..., diag_idx, diag_idx] = 0.0

        inner = torch.diag_embed(grad_eigvals) if grad_eigvals is not None else 0.0
        if grad_eigvecs is not None:
            inner = inner + F * (eigvecs.transpose(-2, -1) @ grad_eigvecs)
        grad_X = eigvecs @ inner @ eigvecs.transpose(-2, -1)
        grad_X = (grad_X + grad_X.transpose(-2, -1)) / 2
        return grad_X, None


def safe_eigh(X, eps=1e-4):
    """Eigendecomposition of a symmetric matrix with a NaN-safe backward (see `_SafeEighFunction`)."""
    return _SafeEighFunction.apply(X, eps)


def log(X):
    """Matrix logarithm of a symmetric matrix."""
    L, U = torch.linalg.eigh(X, UPLO='U')
    L = torch.diag_embed(L.log())
    return U @ L @ U.transpose(-2, -1)


def exp(X):
    """Matrix exponential of a symmetric matrix."""
    L, U = torch.linalg.eigh(X, UPLO='U')
    L = torch.diag_embed(L.exp())
    return U @ L @ U.transpose(-2, -1)


def sqrtm(X):
    """Matrix square root of an SPD matrix."""
    return exp(0.5 * log(X))


def logm(X, Y):
    """Riemannian logarithmic map: tangent vector at X pointing towards Y."""
    C = sqrtm(X)
    C_inv = C.inverse()
    return C @ log(C_inv @ Y @ C_inv) @ C


def expm(X, Y):
    """Riemannian exponential map: walk from X along tangent vector Y."""
    C = sqrtm(X)
    C_inv = C.inverse()
    return C @ exp(C_inv @ Y @ C_inv) @ C


def frechet_mean(spds, num_iter=20):
    """Karcher/Frechet mean of a batch of SPD matrices under the affine-invariant metric."""
    mean = torch.mean(spds, dim=0)
    for _ in range(num_iter):
        c = sqrtm(mean)
        c_inv = c.inverse()
        tangent_mean = log(c_inv @ spds @ c_inv)
        tangent_mean = torch.mean(c @ tangent_mean @ c, dim=0)
        mean = c @ exp(c_inv @ tangent_mean @ c_inv) @ c
    return mean
