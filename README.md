# DeepHoloBrain

Code for

> Tingting Dan, Ziquan Wei, Won Hwa Kim, Guorong Wu. **"Exploring the Enigma
> of Neural Dynamics Through A Scattering-Transform Mixer Landscape for
> Riemannian Manifold."** ICML 2024.

DeepHoloBrain casts manifold-based deep learning for functional-connectivity
(FC) matrices as an MLP-Mixer-style architecture on the Riemannian manifold
of SPD matrices. Instead of learning unconstrained mapping functions, it
builds them from a bank of graph harmonic wavelets derived from each
subject's structural connectivity (SC), giving both a trainable model and a
neuroscience-interpretable coupling between brain structure and function.

## Method overview

| Paper | Code |
|---|---|
| Eq. 1 -- positive mapping `f(X, W) = W^T X W` | `deepholobrain/layers/spd_layers.py::SPDTransform` |
| Sec. 2.3 -- graph harmonic wavelet basis | `deepholobrain/layers/wavelet.py` |
| Eq. 2-3, Fig. 1-3 -- scattering-transform Mixer layer | `deepholobrain/layers/scattering.py::ScatteringMixerLayer` |
| Eq. 4 -- regularization on the scale parameter gamma | `scale_regularization_beta` in `scripts/common.py` |
| Sec. 3.3, Fig. 1/3 -- DeepHoloBrain model | `deepholobrain/models.py::DeepHoloBrain` |
| Sec. 4.2 -- SPDNet comparison baseline | `deepholobrain/models.py::SPDNet` |
| Fig. 4/5 -- node-/frequency-specific attention maps | `attention_row`/`attention_column` returned by `DeepHoloBrain.forward` |
| Sec. 4.1 -- HCP-A / ADNI / OASIS data loaders | `deepholobrain/data/datasets.py` |
| Permutation significance tests (Sec. 4) | `analysis/permutationTest.m` |

`ScatteringMixerLayer` supports two implementations, selected by
`--model_variant` in `scripts/train_task_recognition_hcpa.py`: `optimized`
(default, numerically stabilized) and `paper_theory` (follows Eq. 2-3
literally). Run the script with `--help` for the full flag list.

## Repository layout

```
CODE/
  deepholobrain/           installable package
    layers/                 SPD-manifold layers, harmonic wavelets, the scattering Mixer layer
    optim/                  Riemannian (Stiefel-manifold) optimizer wrapper
    data/                   HCP-A / ADNI / OASIS dataset loaders
    models.py                SPDNet, DeepHoloBrain
    losses.py                 auxiliary manifold losses
  scripts/                  training entry points (argparse CLIs)
    common.py                shared k-fold / holdout train-eval loops
    train_task_recognition_hcpa.py        DeepHoloBrain, HCP-A four-task recognition (Table 1)
    train_task_recognition_hcpa_kfold.py  SPDNet baseline, same task
    train_diagnosis_adni.py               SPDNet baseline, ADNI 5-fold CV (Table 2)
    train_diagnosis_oasis.py              SPDNet baseline, OASIS 5-fold CV (Table 2)
    pretrain_moca_hcpa.py                 DeepHoloBrain, HCP-A MoCA regression pretrain
    pretrain_moca_hcpa_spdnet.py           SPDNet baseline, same pretrain
    finetune_adni.py                      fine-tune a pretrained checkpoint on ADNI
  preprocessing/
    compute_harmonic_wavelets.py          visualize the harmonic wavelet basis of an SC graph
  analysis/
    permutationTest.m                     permutation significance test (Sec. 4)
    summarize_cv.py                       recompute macro-averaged CV metrics from saved fold predictions
    evaluate_checkpoint.py                evaluate one saved fold checkpoint (HCP-A/ADNI/OASIS), save per-class logits
  requirements.txt
```

## Setup

```bash
pip install -r CODE/requirements.txt
```

Requires Python >= 3.9 and PyTorch >= 2.0.

## Data

Each dataset loader in `deepholobrain/data/datasets.py` documents its
expected file layout (FC/SC directory structure, `.mat` key names, label CSV
columns). In brief:

- **HCP-A**: per-subject FC `.csv` (AAL116) and, for DeepHoloBrain, a
  matching per-subject SC `.mat` folder (`aal116_sift_radius2_count_connectivity`).
  Task label (FACENAME/VISMOTOR/CARIT/REST) is inferred from the filename.
- **ADNI**: per-subject FC `.txt` (AAL90), SC `.mat`, and a label CSV with
  `subject_id`, `Label1` columns.
- **OASIS**: per-subject FC `.txt`, SC `.csv`, and a label CSV with
  `SUBJECT_ID`, `Label1` columns.

## Running

Every script takes `--fc_path`/`--sc_path`/`--label_path` for your local
data and `--use_conda` for the torch device string:

```bash
python -m scripts.train_task_recognition_hcpa \
  --fc_path /path/to/HCP-A/FC --sc_path /path/to/HCP-A/SC \
  --spd 1 --use_conda cuda:0

python -m scripts.train_diagnosis_adni \
  --fc_path /path/to/ADNI/FC --sc_path /path/to/ADNI/SC \
  --label_path /path/to/ADNI/label.csv --use_conda cuda:0
```

Run any script with `--help` for its full flag list.

## Citation

```bibtex
@inproceedings{dan2024deepholobrain,
  title     = {Exploring the Enigma of Neural Dynamics Through A Scattering-Transform Mixer Landscape for {R}iemannian Manifold},
  author    = {Dan, Tingting and Wei, Ziquan and Kim, Won Hwa and Wu, Guorong},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024}
}
```
