"""Graph harmonic wavelet basis construction (Sec. 2.3, background to Eq. 2).

Builds the graph Laplacian of a structural-connectivity graph, eigendecomposes
it into the harmonic basis {u_k} with eigenvalues {lambda_k}, and uses that
to build the frequency-scaled filter g_gamma(lambda) = exp(-gamma*lambda) of
Eq. 2. Three constructions:

- `graph_harmonic_basis`: symmetric-normalized Laplacian, per-frequency
  rank-1 contribution lambda_k * u_k u_k^T (used when `deepholobrain=False`).
- `graph_harmonic_components` / `deepholobrain_harmonic_wavelet_matrix`: apply
  `exp(-gamma*lambda_k)` per-frequency before forming u_k u_k^T, matching
  Eq. 2's literal `psi_ik = U g_k(gamma*Lambda) U^T delta_i` (used when
  `deepholobrain=True`).
- `graph_harmonic_basis_legacy`: reproduces the original ICML 2024 code
  release's basis exactly, for reproduction only.
"""

import torch

from .riemannian import safe_eigh


def _symmetric_normalized_laplacian(sc):
    node_num = sc.shape[-1]
    degree = sc.sum(dim=2)
    has_degree = degree > 0
    safe_degree = torch.where(has_degree, degree, torch.ones_like(degree))
    inv_sqrt_degree = torch.where(has_degree, 1.0 / torch.sqrt(safe_degree), torch.zeros_like(degree))

    laplacian = -sc.clone()
    laplacian = inv_sqrt_degree.unsqueeze(2) * laplacian * inv_sqrt_degree.unsqueeze(1)
    diag_idx = torch.arange(node_num, device=sc.device)
    laplacian[:, diag_idx, diag_idx] = 1.0
    return (laplacian + laplacian.transpose(1, 2)) / 2


def _harmonic_eigendecomposition(sc, eigh_eps):
    if sc.dim() == 2:
        sc = sc.unsqueeze(0)
    eigvals, eigvecs = safe_eigh(_symmetric_normalized_laplacian(sc), eigh_eps)
    eigvals = torch.clamp(eigvals, min=0)
    signs = torch.sign(eigvecs[:, 0, :])
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    eigvecs = eigvecs * signs.unsqueeze(1)
    return eigvals, eigvecs


def graph_harmonic_basis(sc, eigh_eps=1e-4):
    """Per-frequency harmonic basis: rank-1 matrix lambda_k * u_k u_k^T per frequency k.

    Returns (B, N, N*N): N rank-1 N x N matrices (flattened) per batch element.
    """
    eigvals, eigvecs = _harmonic_eigendecomposition(sc, eigh_eps)
    outer = torch.einsum('bik,bjk->bkij', eigvecs, eigvecs)
    basis = eigvals.unsqueeze(-1).unsqueeze(-1) * outer
    return basis.reshape(basis.size(0), basis.size(1), -1)


def graph_harmonic_components(sc, eigh_eps=1e-4):
    """Laplacian frequencies and rank-one Fourier projectors, kept separate."""
    eigvals, eigvecs = _harmonic_eigendecomposition(sc, eigh_eps)
    projectors = torch.einsum('bik,bjk->bkij', eigvecs, eigvecs)
    return eigvals, projectors


def deepholobrain_harmonic_wavelet_matrix(sc, gamma, num_frequencies=None, eigh_eps=1e-4):
    """Block wavelet matrix P from Eq. 2: psi_ik[a] = exp(-gamma*lambda_k) * u_k[a] * u_k[i].

    Uses the combinatorial Laplacian L = D - A. Columns ordered by location i
    then frequency k, so P.shape == (B, N, N*K).
    """
    if sc.dim() == 2:
        sc = sc.unsqueeze(0)
    adjacency = torch.clamp((sc + sc.transpose(-2, -1)) / 2, min=0)
    adjacency = adjacency - torch.diag_embed(adjacency.diagonal(dim1=-2, dim2=-1))
    degree = adjacency.sum(dim=-1)
    laplacian = torch.diag_embed(degree) - adjacency
    eigvals, eigvecs = safe_eigh(laplacian, eigh_eps)
    eigvals = torch.clamp(eigvals, min=0)

    k = eigvals.shape[-1] if num_frequencies is None else num_frequencies
    if not 1 <= k <= eigvals.shape[-1]:
        raise ValueError(f'num_frequencies must be in [1, {eigvals.shape[-1]}], got {k}')
    eigvals = eigvals[:, :k]
    eigvecs = eigvecs[:, :, :k]
    gain = torch.exp(-gamma * eigvals)
    wavelets = torch.einsum('bak,bik,bk->baik', eigvecs, eigvecs, gain)
    return wavelets.reshape(wavelets.shape[0], wavelets.shape[1], -1), eigvals


def _symmetric_normalized_laplacian_legacy(sc):
    node_num = sc.shape[-1]
    degree = sc.sum(dim=2)
    inv_sqrt_degree = torch.where(degree > 0, 1.0 / torch.sqrt(degree), torch.zeros_like(degree))

    laplacian = -sc.clone()
    laplacian += torch.eye(node_num, device=sc.device, dtype=sc.dtype).expand_as(laplacian)
    laplacian = inv_sqrt_degree.unsqueeze(2) * laplacian * inv_sqrt_degree.unsqueeze(1)
    return (laplacian + laplacian.transpose(1, 2)) / 2


def graph_harmonic_basis_legacy(sc):
    """Original ICML 2024 code release's harmonic basis, reproduced exactly."""
    if sc.dim() == 2:
        sc = sc.unsqueeze(0)

    node_num = sc.shape[-1]
    sym_laplacian = _symmetric_normalized_laplacian_legacy(sc)
    eigvecs, eigvals, _ = torch.linalg.svd(sym_laplacian)
    eigvals = torch.clamp(eigvals, min=0)

    diag = torch.diag_embed(eigvals).unsqueeze(0).repeat(node_num, 1, 1, 1)
    per_frequency = torch.einsum("bjk,nbkl,bli->nbji", eigvecs, diag, eigvecs.transpose(1, 2))
    basis = per_frequency.view(per_frequency.size(0), per_frequency.size(1), -1)
    return basis.permute(1, 0, 2)


create_normalized_matrices = graph_harmonic_basis
