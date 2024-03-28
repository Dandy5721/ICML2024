"""Compute and visualize the per-frequency graph harmonic wavelet basis of a
structural-connectivity graph (background to Eq. 2, Sec. 2.3).

Reference, unbatched, per-index implementation: for each frequency index k,
builds the rank-1 matrix lambda_k * u_k u_k^T from the symmetric-normalized
graph Laplacian's eigendecomposition.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def graph_harmonic_wavelets(sc):
    """Per-frequency rank-1 harmonic wavelet matrices of an (N, N) SC graph."""
    node_num = sc.shape[0]
    degree = np.sum(sc, axis=1)
    has_degree = degree > 0

    laplacian = -sc.copy()
    laplacian[has_degree, :] /= np.sqrt(degree[has_degree, None])
    laplacian[:, has_degree] /= np.sqrt(degree[None, has_degree])
    np.fill_diagonal(laplacian, 1)

    sym_laplacian = (laplacian + laplacian.T) / 2
    eigvecs, eigvals, _ = np.linalg.svd(sym_laplacian)
    eigvals = np.sort(eigvals)
    order = np.argsort(np.linalg.svd(sym_laplacian)[1])
    eigvecs = eigvecs[:, order]
    signs = np.sign(eigvecs[0, :])
    signs[signs == 0] = 1
    eigvecs = eigvecs @ np.diag(signs)

    wavelets = np.zeros((node_num, node_num, node_num))
    for k in range(node_num):
        diag = np.zeros((node_num, node_num))
        diag[k, k] = eigvals[k]
        wavelets[k] = eigvecs @ diag @ eigvecs.T
    return wavelets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sc_mat', required=True, help='.mat file with an SC_avg56-style N x N matrix.')
    parser.add_argument('--mat_key', default='SC_avg56', help="Key of the SC matrix inside --sc_mat.")
    parser.add_argument('--out_dir', required=True, help='Directory to write matrix_<k>.png visualizations to.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sc = scipy.io.loadmat(args.sc_mat)[args.mat_key]
    wavelets = graph_harmonic_wavelets(sc)

    for k, wavelet_k in enumerate(wavelets):
        print(f'frequency {k}: max={wavelet_k.max():.4f} mean={wavelet_k.mean():.4f}')
        plt.imshow(wavelet_k, cmap='PuRd')
        plt.colorbar()
        plt.savefig(os.path.join(args.out_dir, f'matrix_{k + 1}.png'))
        plt.close()

    print(f'wavelets shape: {wavelets.shape}')
