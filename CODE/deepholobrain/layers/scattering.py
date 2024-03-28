"""The scattering-transform Mixer layer (Sec. 3.1-3.2, Eq. 2-3, Fig. 1-3).

Builds the block matrix of graph harmonic wavelets P from the SC graph,
applies the frequency scaling g_gamma(lambda) = exp(-gamma*lambda) of Eq. 2,
weights it with node-/frequency-specific attention (Fig. 4/5), forms the
Supra-FC matrix P^T X P (Eq. 3), and max-pools to an N x N SPD matrix
(Prop. 3.6).

"""

import torch
from torch import nn
from torch.nn import functional as F

from ..utils import fc2vector
from .riemannian import safe_eigh
from .wavelet import (
    graph_harmonic_basis,
    graph_harmonic_basis_legacy,
    graph_harmonic_components,
    deepholobrain_harmonic_wavelet_matrix,
)


class ScatteringMixerLayer(nn.Module):
    """Row/column graph-scattering mapping P^T X P with location/frequency attention."""

    def __init__(self, num_nodes=116, use_legacy_wavelet=False, residual_fc=True,
                 gated_scattering=False, normalize_scattering_branch=False,
                 nonnegative_scattering_gate=False, spectral_filter_wavelet=False,
                 deepholobrain=False, mlp_mixer=False, gamma_init=1.0,
                 location_frequency_attention=False, num_frequencies=None,
                 eq3_ln_relu=False, eq3_norm='none',
                 eq3_activation='none'):
        super(ScatteringMixerLayer, self).__init__()
        self.num_nodes = num_nodes
        self.residual_fc = residual_fc
        self.gated_scattering = gated_scattering
        self.normalize_scattering_branch = normalize_scattering_branch
        self.nonnegative_scattering_gate = nonnegative_scattering_gate
        self.spectral_filter_wavelet = spectral_filter_wavelet
        self.deepholobrain = deepholobrain
        self.mlp_mixer = mlp_mixer
        self.location_frequency_attention = location_frequency_attention
        self.eq3_ln_relu = eq3_ln_relu
        if eq3_norm not in ('none', 'column', 'row', 'both'):
            raise ValueError(f'unsupported eq3_norm={eq3_norm!r}')
        if eq3_activation not in ('none', 'relu'):
            raise ValueError(f'unsupported eq3_activation={eq3_activation!r}')
        self.eq3_norm = eq3_norm
        self.eq3_activation = eq3_activation
        self.num_frequencies = num_nodes if num_frequencies is None else int(num_frequencies)
        if not 1 <= self.num_frequencies <= num_nodes:
            raise ValueError(f'num_frequencies must be in [1, {num_nodes}], got {self.num_frequencies}')
        if deepholobrain and residual_fc:
            raise ValueError('the deepholobrain variant follows Eq. 2-3 and therefore forbids an FC residual')
        if gated_scattering:
            initial_gain = torch.full((1,), -6.0) if nonnegative_scattering_gate else torch.zeros(1)
            self.scattering_gain = nn.Parameter(initial_gain)
        else:
            self.scattering_gain = None
        self._wavelet_fn = graph_harmonic_basis_legacy if use_legacy_wavelet else graph_harmonic_basis
        self.scale = nn.Parameter(
            torch.full((1,), float(gamma_init)) if deepholobrain else torch.randn(1)
        )
        attention_width = num_nodes * (self.num_frequencies if deepholobrain else num_nodes)
        self.attention_row = nn.Parameter(torch.randn(1, 1, attention_width))
        self.attention_column = nn.Parameter(torch.randn(1, num_nodes, 1))
        pool_size = self.num_frequencies if deepholobrain else num_nodes
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        if deepholobrain and mlp_mixer:
            self.frequency_norm = nn.LayerNorm(self.num_frequencies)
            self.frequency_mlp = nn.Linear(self.num_frequencies, self.num_frequencies)
            self.location_norm = nn.LayerNorm(num_nodes)
            self.location_mlp = nn.Linear(num_nodes, num_nodes)
            self.mixer_gain = nn.Parameter(torch.tensor(-2.0))

    def forward(self, input, sc, attention_mode):
        if self.deepholobrain:
            weighted_matrix, _ = deepholobrain_harmonic_wavelet_matrix(
                sc, self.scale, num_frequencies=self.num_frequencies,
            )
            if self.mlp_mixer:
                bank = weighted_matrix.reshape(
                    weighted_matrix.shape[0], self.num_nodes, self.num_nodes, self.num_frequencies,
                )
                mixed_frequency = F.relu(self.frequency_mlp(self.frequency_norm(bank)))
                mixed_location = bank.transpose(-2, -1)
                mixed_location = F.relu(self.location_mlp(self.location_norm(mixed_location)))
                mixed_location = mixed_location.transpose(-2, -1)
                mix = torch.sigmoid(self.mixer_gain)
                bank = bank + mix * (mixed_frequency + mixed_location)
                weighted_matrix = bank.reshape(bank.shape[0], self.num_nodes, -1)

            if self.location_frequency_attention:
                attention_width = self.num_nodes * self.num_frequencies
                location_frequency = torch.softmax(self.attention_row, dim=-1) * attention_width
                input_region = torch.softmax(self.attention_column, dim=1) * self.num_nodes
            else:
                location_frequency = torch.ones_like(self.attention_row)
                input_region = torch.ones_like(self.attention_column)
            weighted_matrix = weighted_matrix * location_frequency * input_region
            attention_rows = (location_frequency / (self.num_nodes * self.num_frequencies)).squeeze(0)
            attention_columns = (input_region / self.num_nodes).squeeze(0)
        elif self.spectral_filter_wavelet:
            eigvals, projectors = graph_harmonic_components(sc)
            spectral_gain = torch.exp(-self.scale * eigvals).unsqueeze(-1).unsqueeze(-1)
            scaled_wavelet = (spectral_gain * projectors).reshape(
                projectors.shape[0], projectors.shape[1], -1
            )
        else:
            harmonic_basis = self._wavelet_fn(sc)
            harmonic_basis = harmonic_basis / (
                harmonic_basis.detach().abs().amax(dim=(1, 2), keepdim=True) + 1e-8
            )
            scaled_wavelet = torch.exp(-self.scale * harmonic_basis)

        if not self.deepholobrain:
            attention_rows = torch.softmax(self.attention_row, dim=2)
            attention_columns = torch.softmax(self.attention_column, dim=1)

        if self.deepholobrain:
            pass
        elif attention_mode == 0:
            weighted_matrix = scaled_wavelet * self.attention_row
            weighted_matrix = weighted_matrix * self.attention_column
            attention_rows = attention_rows.squeeze(0)
            attention_columns = attention_columns.squeeze(0)
        else:
            attention_rows = attention_rows.view(1, 1, -1)
            weighted_matrix = scaled_wavelet * attention_rows
            weighted_matrix = weighted_matrix * attention_columns
            attention_rows = attention_rows.squeeze(0)
            attention_columns = attention_columns.squeeze(0)

        # Eq. 3: column-wise then row-wise scattering, P^T X followed by (P^T X) P.
        column_mixed = weighted_matrix.transpose(-2, -1) @ input
        column_norm = self.eq3_ln_relu or self.eq3_norm in ('column', 'both')
        row_norm = self.eq3_ln_relu or self.eq3_norm in ('row', 'both')
        use_relu = self.eq3_ln_relu or self.eq3_activation == 'relu'
        if self.deepholobrain and column_norm:
            column_mixed = F.layer_norm(column_mixed, (column_mixed.shape[-1],))
        if self.deepholobrain and use_relu:
            column_mixed = F.relu(column_mixed)
        output = column_mixed @ weighted_matrix
        if self.deepholobrain and row_norm:
            output = F.layer_norm(output, (output.shape[-1],))
        if self.deepholobrain and use_relu:
            output = F.relu(output)

        if output.dim() == 2:
            output = output.unsqueeze(0)
        elif output.dim() != 3:
            raise RuntimeError(f"Unexpected output shape for pooling: {tuple(output.shape)}")

        pooled_output = self.pool(output)
        if self.deepholobrain:
            # Eq. 3 symmetrizes after pooling; Prop. 3.6's positive-definiteness
            # can fail at machine precision, so rectify at a relative 1e-6 floor.
            pooled_output = (pooled_output + pooled_output.transpose(-2, -1)) / 2
            eigenvalues, eigenvectors = safe_eigh(pooled_output, 1e-6)
            matrix_scale = eigenvalues.abs().mean(-1, keepdim=True).clamp_min(1e-8)
            eigenvalues = eigenvalues.clamp_min(1e-6 * matrix_scale)
            pooled_output = eigenvectors @ torch.diag_embed(eigenvalues) @ eigenvectors.transpose(-2, -1)
        if self.residual_fc:
            residual = input if input.dim() == 3 else input.unsqueeze(0)
            if self.normalize_scattering_branch:
                scatter_trace = pooled_output.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
                residual_trace = residual.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
                pooled_output = pooled_output * (residual_trace / (scatter_trace + 1e-12))
            if self.scattering_gain is not None:
                gain = F.softplus(self.scattering_gain) if self.nonnegative_scattering_gate else self.scattering_gain
                pooled_output = gain * pooled_output + residual
            else:
                pooled_output = pooled_output + residual
        if attention_mode == 0:
            pooled_output = fc2vector(pooled_output)

        return pooled_output, self.scale, attention_rows, attention_columns
