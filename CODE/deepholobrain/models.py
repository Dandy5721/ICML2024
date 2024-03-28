"""Models evaluated in the paper: the SPDNet baseline (Sec. 4, Huang & Van Gool 2017)
and DeepHoloBrain, the proposed scattering-transform Mixer (Sec. 3.3, Fig. 1/3).
"""

import torch
from torch import nn

from .layers.scattering import ScatteringMixerLayer
from .layers.spd_layers import Normalize, SPDRectified, SPDTangentSpace, SPDTransform


class SPDNet(nn.Module):
    """SPDNet baseline (Huang & Van Gool, 2017): a stack of Eq. 1 positive mappings.

    `spd` selects the network depth (0-4 use progressively deeper/shallower
    layer stacks; anything else falls back to `layers`/`classifier`),
    matching the `--spd` ablation flag used across the training scripts.
    """

    def __init__(self, num_classes):
        super(SPDNet, self).__init__()
        self.layers = nn.Sequential(
            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTransform(64, 32, 1),
            SPDRectified(),
            SPDTransform(32, 16, 1),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier = nn.Sequential(nn.Linear(16 * 17 // 2, num_classes))

        self.layers0 = nn.Sequential(
            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTransform(64, 32, 1),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier0 = nn.Sequential(nn.Linear(32 * 33 // 2, num_classes))

        self.layers1 = nn.Sequential(
            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier1 = nn.Sequential(nn.Linear(64 * 65 // 2, num_classes))

        self.layers2 = nn.Sequential(
            SPDTransform(116, 64, 1),
            SPDRectified(),
            SPDTransform(64, 32, 1),
            SPDRectified(),
            SPDTransform(32, 16, 1),
            SPDRectified(),
            SPDTransform(16, 8, 1),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier2 = nn.Sequential(nn.Linear(8 * 9 // 2, num_classes))

        self.layers3 = nn.Sequential(
            SPDTransform(116, 16, 1),
            SPDRectified(),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier3 = nn.Sequential(nn.Linear(16 * 17 // 2, num_classes))

        self.layers4 = nn.Sequential(
            SPDTransform(116, 32, 1),
            SPDRectified(),
            SPDTangentSpace(vectorize_all=True),
            Normalize(),
        )
        self.classifier4 = nn.Sequential(nn.Linear(32 * 33 // 2, num_classes))

    def forward(self, x, spd):
        layers, classifier = {
            0: (self.layers0, self.classifier0),
            1: (self.layers1, self.classifier1),
            2: (self.layers2, self.classifier2),
            3: (self.layers3, self.classifier3),
            4: (self.layers4, self.classifier4),
        }.get(spd, (self.layers, self.classifier))
        x = layers(x)
        out = classifier(x)
        return x, out


class DeepHoloBrain(nn.Module):
    """DeepHoloBrain (Sec. 3.3): scattering-transform Mixer for the SPD manifold.

    Computes the Supra-FC pooling of Eq. 2-3 via `ScatteringMixerLayer`, then
    `spd == 1` feeds it through a manifold stack (`self.layers`) into
    `classifier1` (main path, Fig. 1/3), or `spd == 0` feeds it directly to
    the plain-MLP `classifier2` ablation.

    Returns `(scale, out, attention_rows, attention_columns)`: `scale` is
    Eq. 2's gamma (regularized via Eq. 4), `attention_rows`/`attention_columns`
    are the Fig. 4/5 attention maps.
    """

    def __init__(self, num_classes, num_nodes=116, use_legacy_wavelet=False, residual_fc=True,
                 gated_scattering=False, manifold_dim=16, normalize_scattering_branch=False,
                 nonnegative_scattering_gate=False, classifier_hidden_dim=0,
                 final_rectify=False, spectral_filter_wavelet=False,
                 deepholobrain=False, mlp_mixer=False, gamma_init=1.0,
                 location_frequency_attention=False, num_frequencies=None, classifier_dropout=0.0,
                 eq3_ln_relu=False, eq3_norm='none',
                 eq3_activation='none'):
        super(DeepHoloBrain, self).__init__()
        if manifold_dim == 16:
            self.layers = nn.Sequential(
                SPDTransform(num_nodes, 64, 1), SPDRectified(),
                SPDTransform(64, 32, 1), SPDRectified(),
                SPDTransform(32, 16, 1),
                *([SPDRectified()] if final_rectify else []),
                SPDTangentSpace(vectorize_all=False), Normalize(),
            )
        elif manifold_dim == 32:
            self.layers = nn.Sequential(
                SPDTransform(num_nodes, 64, 1), SPDRectified(),
                SPDTransform(64, 32, 1),
                *([SPDRectified()] if final_rectify else []),
                SPDTangentSpace(vectorize_all=False), Normalize(),
            )
        elif manifold_dim == 64:
            self.layers = nn.Sequential(
                SPDTransform(num_nodes, 64, 1),
                *([SPDRectified()] if final_rectify else []),
                SPDTangentSpace(vectorize_all=False), Normalize(),
            )
        else:
            raise ValueError(f'manifold_dim must be 16, 32, or 64, got {manifold_dim}')
        # Attribute kept as `weighted_scaled` (rather than e.g. `scattering`)
        # so state_dicts saved from the original code release stay loadable.
        self.weighted_scaled = ScatteringMixerLayer(
            num_nodes=num_nodes,
            use_legacy_wavelet=use_legacy_wavelet,
            residual_fc=residual_fc,
            gated_scattering=gated_scattering,
            normalize_scattering_branch=normalize_scattering_branch,
            nonnegative_scattering_gate=nonnegative_scattering_gate,
            spectral_filter_wavelet=spectral_filter_wavelet,
            deepholobrain=deepholobrain,
            mlp_mixer=mlp_mixer,
            gamma_init=gamma_init,
            location_frequency_attention=location_frequency_attention,
            num_frequencies=num_frequencies,
            eq3_ln_relu=eq3_ln_relu,
            eq3_norm=eq3_norm,
            eq3_activation=eq3_activation,
        )
        tangent_dim = manifold_dim * (manifold_dim + 1) // 2
        if not 0.0 <= classifier_dropout < 1.0:
            raise ValueError('classifier_dropout must be in [0, 1)')
        if classifier_hidden_dim > 0:
            self.classifier1 = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(tangent_dim, classifier_hidden_dim),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(classifier_hidden_dim, num_classes),
            )
        else:
            self.classifier1 = nn.Sequential(
                nn.Dropout(classifier_dropout),
                nn.Linear(tangent_dim, num_classes),
            )
        self.classifier2 = nn.Sequential(
            nn.Linear(num_nodes * (num_nodes - 1) // 2, 1280),
            nn.Linear(1280, 640),
            nn.Linear(640, 320),
            nn.Linear(320, 16 * 17 // 2),
            nn.Linear(16 * 17 // 2, num_classes),
        )

    def forward(self, x, sc, spd):
        x, scale, attention_rows, attention_columns = self.weighted_scaled(x, sc, spd)

        if spd == 1:
            # Rescale to a fixed average eigenvalue (trace/N) before the manifold stack.
            trace = x.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
            x = x / (trace / x.shape[-1] + 1e-12)
            x = self.layers(x)
            out = self.classifier1(x)
        else:
            out = self.classifier2(x)

        return scale, out, attention_rows, attention_columns
