"""Compare a symmetry-projected two-state NOCI with five LF-HF states."""

import numpy as np
import pytest
from scipy.linalg import eigh

from src import lf_mp, lf_mr, my_direct_ep


def test_alpha_2p2_two_state_projection_matches_five_state_ground_energy():
    omega = 0.5
    tmat = my_direct_ep.electron_ring_hopping(4, -1.0)
    settings = dict(nrandom=8, random_scale=0.5, seed=0, gtol=1e-8, max_cycle=500)
    symmetric, results22 = lf_mp.lf_hf_full_alpha_point(
        2.2, tmat, omega, **settings
    )
    broken, _ = lf_mp.lf_hf_full_alpha_point(2.4, tmat, omega, **settings)
    assert symmetric.density_imbalance < 1e-8
    assert broken.density_imbalance > 0.6

    local_lam, local_shift = lf_mr.lf_translation_orbit(
        broken.lam, broken.shift
    )
    frame_lam = np.concatenate((symmetric.lam[None], local_lam))
    frame_shift = np.concatenate((symmetric.shift[None], local_shift))
    # Each row is one complete LF-HF wavefunction's site coefficients.
    orbital = np.stack(
        [symmetric.coeff]
        + [np.roll(broken.coeff, R) for R in range(4)]
    )
    g22 = my_direct_ep.alpha_to_g(2.2, omega)
    skernel = lf_mr.lf_frame_overlap(frame_lam, frame_shift)
    hkernel = lf_mr.lf_frame_hamiltonian(
        tmat, g22, omega, frame_lam, frame_shift
    )
    s5 = np.einsum("ip,ipjq,jq->ij", orbital, skernel, orbital)
    h5 = np.einsum("ip,ipjq,jq->ij", orbital, hkernel, orbital)
    e5 = eigh(h5, s5, eigvals_only=True)[0]

    # The four translated states are not mutually orthogonal, so their
    # 1/2-weighted sum still requires normalization through S.
    projection = np.zeros((5, 2))
    projection[0, 0] = 1.0
    projection[1:, 1] = 0.5
    s2 = projection.T @ s5 @ projection
    h2 = projection.T @ h5 @ projection
    e2, vectors = eigh(h2, s2)
    residual = np.linalg.norm(
        h2 @ vectors[:, 0] - e2[0] * s2 @ vectors[:, 0]
    )
    assert s2[1, 1] == pytest.approx(1.49931887, abs=1e-8)
    assert residual < 1e-12
    assert e2[0] == pytest.approx(-2.930178129612, abs=1e-10)
    assert e2[0] == pytest.approx(e5, abs=1e-12)

    sym_lam, sym_shift = lf_mr.lf_translation_orbit(
        symmetric.lam, symmetric.shift
    )
    cs_lam, cs_shift = lf_mr.lf_translation_orbit(
        results22[0].lam, results22[0].shift
    )
    full_lam = np.concatenate((sym_lam, cs_lam, local_lam))
    full_shift = np.concatenate((sym_shift, cs_shift, local_shift))
    full_overlap = lf_mr.lf_frame_overlap(full_lam, full_shift)
    full_hamiltonian = lf_mr.lf_frame_hamiltonian(
        tmat, g22, omega, full_lam, full_shift
    )
    full_energy, _, rank = lf_mr.lf_noci_lowest(
        full_hamiltonian, full_overlap
    )
    assert rank == 20
    assert full_energy == pytest.approx(-2.930809431151, abs=1e-10)
    assert e2[0] - full_energy == pytest.approx(0.000631301539, abs=1e-10)
