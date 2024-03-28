"""HCP-A four-task recognition with DeepHoloBrain (Sec. 4.2, Table 1).

Five-fold cross-validation over FACENAME, VISMOTOR, CARIT and REST. See
`--model_variant`, `--enable_mlp_mixer`, `--enable_location_frequency_attention` and
`--eq3_*` for the literal-Eq.2-3 (`deepholobrain`) path versus the default,
numerically-stabilized `optimized` path (see
`deepholobrain.layers.scattering.ScatteringMixerLayer`).
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepholobrain.data import FC_SCDataset
from deepholobrain.models import DeepHoloBrain
from scripts.common import run_kfold_classification


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fc_path', required=True, help='Dir of per-subject FC .csv files (AAL116).')
    parser.add_argument('--sc_path', required=True, help='Dir of per-subject SC .mat folders.')
    parser.add_argument('--output_path', type=str, default='hcpa_deepholobrain')
    parser.add_argument('--model_variant', choices=('optimized', 'deepholobrain'), default='optimized',
                        help='deepholobrain follows Eq. 2-3 literally, without an FC residual.')
    parser.add_argument('--enable_mlp_mixer', action='store_true',
                        help='Enable a legacy unconstrained Linear-layer ablation; P^T X P already implements the Eq. 3 scattering Mixer.')
    parser.add_argument('--enable_location_frequency_attention', action='store_true',
                        help='Enable additional learned attention on P (off in the pure Eq. 2 default).')
    parser.add_argument('--eq3_ln_relu', action='store_true',
                        help='Legacy alias for --eq3_norm both --eq3_activation relu.')
    parser.add_argument('--eq3_norm', choices=('none', 'column', 'row', 'both'), default='none',
                        help='Optional parameter-free LayerNorm placement in the Eq. 3 Mixer.')
    parser.add_argument('--eq3_activation', choices=('none', 'relu'), default='none',
                        help='Optional activation after both axis-wise Eq. 3 mappings.')
    parser.add_argument('--gamma_init', type=float, default=1.0,
                        help='Initial gamma in paper Eq. 2 (paper does not report its value).')
    parser.add_argument('--num_frequencies', type=int, default=None,
                        help='Number K of lowest graph frequencies (paper specifies K <= N).')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--number_cls', type=int, default=4)
    parser.add_argument('--spd', type=int, default=1, help='1: paper manifold path; 0: plain-MLP ablation.')
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0,
                        help='0 runs all folds; 1..num_folds runs one fold (for parallel evaluation).')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--scale_regularization_beta', type=float, default=1.0,
                        help='Beta in Eq. 4 (the paper does not report its numerical value).')
    parser.add_argument('--class_weighting', choices=('none', 'balanced', 'sqrt_balanced'), default='none',
                        help='Optional inverse-frequency CE weighting; paper baseline uses none.')
    parser.add_argument('--class_weight_power', type=float, default=None,
                        help='Optional exponent in [0,1] applied to inverse-frequency weights.')
    parser.add_argument('--focal_gamma', type=float, default=0.0,
                        help='Use focal cross-entropy with this gamma when greater than zero.')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Cross-entropy label smoothing in [0,1); ignored by focal loss.')
    parser.add_argument('--optimizer', choices=('sgd', 'adamw'), default='sgd',
                        help='Base optimizer wrapped by the Riemannian projection/retraction step.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Optimizer weight decay (paper reports 1e-5).')
    parser.add_argument('--lr_scheduler', choices=('none', 'cosine'), default='none',
                        help='Optional learning-rate schedule; paper baseline uses none.')
    parser.add_argument('--gated_scattering', action='store_true',
                        help='Zero-initialize a learnable gate on the scattering branch.')
    parser.add_argument('--manifold_dim', type=int, choices=(16, 32, 64), default=16,
                        help='Final SPD/tangent dimension; released DeepHoloBrain uses 16.')
    parser.add_argument('--classifier_hidden_dim', type=int, default=0,
                        help='Optional hidden width for a nonlinear tangent-space classifier head.')
    parser.add_argument('--classifier_dropout', type=float, default=0.0,
                        help='Dropout on tangent-space classifier features.')
    parser.add_argument('--final_rectify', action='store_true',
                        help='Rectify eigenvalues after the final SPD projection, as in SPDNet.')
    parser.add_argument('--normalize_scattering_branch', action='store_true',
                        help='Trace-align scattering and FC branches before gated fusion.')
    parser.add_argument('--nonnegative_scattering_gate', action='store_true',
                        help='Use a softplus gate so gated FC+scattering fusion remains SPD.')
    parser.add_argument('--spectral_filter_wavelet', action='store_true',
                        help='Apply Eq. 2 scaling to eigenvalues before harmonic projectors.')
    parser.add_argument('--resume', action='store_true',
                        help='Resume each selected fold from its latest saved checkpoint.')
    parser.add_argument('--use_conda', type=str, default='cuda:0', help='torch device string.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.use_conda if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    deepholobrain = args.model_variant == 'deepholobrain'
    if deepholobrain:
        incompatible = []
        if args.gated_scattering:
            incompatible.append('--gated_scattering')
        if args.normalize_scattering_branch:
            incompatible.append('--normalize_scattering_branch')
        if args.nonnegative_scattering_gate:
            incompatible.append('--nonnegative_scattering_gate')
        if args.spectral_filter_wavelet:
            incompatible.append('--spectral_filter_wavelet (already intrinsic to --model_variant deepholobrain)')
        if incompatible:
            raise ValueError('--model_variant deepholobrain does not accept: ' + ', '.join(incompatible))

    # Paper Sec. 4.1 divides by each subject's total fiber count, preserving
    # SC symmetry. The optimized path keeps historical row normalization so
    # completed experiments remain reproducible.
    dataset = FC_SCDataset(
        args.fc_path, args.sc_path,
        sc_normalization='total' if deepholobrain else 'row',
    )
    print(f'num_sample={len(dataset)} num_folds={args.num_folds}')

    config_dir = os.path.join('test_results', args.output_path)
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, 'run_config.json'), 'w', encoding='utf-8') as handle:
        json.dump({
            **vars(args),
            'deepholobrain': deepholobrain,
            'sc_normalization_in_loader': 'total' if deepholobrain else 'row',
            'fc_residual': not deepholobrain,
            'laplacian': 'combinatorial_D_minus_A' if deepholobrain else 'symmetric_normalized',
        }, handle, indent=2, sort_keys=True)

    run_kfold_classification(
        dataset=dataset,
        model_fn=lambda num_classes: DeepHoloBrain(
            num_classes, gated_scattering=args.gated_scattering,
            manifold_dim=args.manifold_dim,
            normalize_scattering_branch=args.normalize_scattering_branch,
            nonnegative_scattering_gate=args.nonnegative_scattering_gate,
            classifier_hidden_dim=args.classifier_hidden_dim,
            final_rectify=args.final_rectify,
            spectral_filter_wavelet=args.spectral_filter_wavelet,
            residual_fc=not deepholobrain,
            deepholobrain=deepholobrain,
            mlp_mixer=args.enable_mlp_mixer,
            gamma_init=args.gamma_init,
            location_frequency_attention=args.enable_location_frequency_attention,
            num_frequencies=args.num_frequencies,
            eq3_ln_relu=args.eq3_ln_relu,
            eq3_norm=args.eq3_norm,
            eq3_activation=args.eq3_activation,
            classifier_dropout=args.classifier_dropout,
        ),
        num_classes=args.number_cls,
        num_folds=args.num_folds, epochs=args.epochs, lr=args.lr, spd=args.spd,
        device=device, output_dir_name=args.output_path, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
        scale_regularization_beta=args.scale_regularization_beta,
        only_fold=args.fold or None,
        class_weighting=args.class_weighting,
        class_weight_power=args.class_weight_power,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        resume=args.resume,
    )
