"""DeepHoloBrain: a scattering-transform Mixer for the Riemannian manifold of SPD matrices.

Dan, Wei, Kim & Wu. "Exploring the Enigma of Neural Dynamics Through a
Scattering-Transform Mixer Landscape for Riemannian Manifold." ICML 2024.

    layers/   SPD-manifold layers (Sec. 2.2), graph harmonic wavelets
              (Sec. 2.3 / Eq. 2), and the scattering-transform Mixer layer
              (Sec. 3.1-3.2 / Eq. 1-3).
    optim/    Riemannian (Stiefel-manifold) optimizer wrapper.
    data/     HCP-A / ADNI / OASIS dataset loaders (Sec. 4.1).
    models.py SPDNet (baseline) and DeepHoloBrain (Sec. 3.3).
    losses.py Auxiliary manifold losses.
"""

__version__ = "1.0.0"
