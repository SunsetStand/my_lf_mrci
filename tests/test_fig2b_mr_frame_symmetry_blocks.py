"""Check translation and reflection reduction of the full MR-LF frame span."""

import numpy as np
import pytest
from scipy.linalg import eigh

from src import lf_mp, lf_mr, my_direct_ep


def test_alpha_2p2_full_frame_ground_state_lives_in_small_symmetry_block():
    length = 4
    omega = 0.5
    tmat = my_direct_ep.electron_ring_hopping(length, -1.0)
    settings = dict(nrandom=8, random_scale=0.5, seed=0, gtol=1e-8, max_cycle=500)
    symmetric, results22 = lf_mp.lf_hf_full_alpha_point(
        2.2, tmat, omega, **settings
    )
    broken, _ = lf_mp.lf_hf_full_alpha_point(
        2.4, tmat, omega, **settings
    )
    g22 = my_direct_ep.alpha_to_g(2.2, omega)
    local_lam, local_shift = lf_mr.lf_translation_orbit(
        broken.lam, broken.shift
    )
    frame_lam = np.concatenate((symmetric.lam[None], local_lam))
    frame_shift = np.concatenate((symmetric.shift[None], local_shift))
    smat = lf_mr.lf_frame_overlap(frame_lam, frame_shift)
    hmat = lf_mr.lf_frame_hamiltonian(
        tmat, g22, omega, frame_lam, frame_shift
    )
    full_energy, _, rank = lf_mr.lf_noci_lowest(hmat, smat)
    assert rank == 20
    s20 = smat.reshape(20, 20)
    h20 = hmat.reshape(20, 20)

    # Translation sends |R,p> to |R+1,p+1>. At each momentum k,
    # one symmetric-frame channel and four relative-site channels remain.
    blocks = []
    energies = []
    for m in range(length):
        k = 2 * np.pi * m / length
        transform = np.zeros((20, 5), dtype=np.complex128)
        for p in range(length):
            transform[p, 0] = np.exp(1j * k * p) / 2
        for relative_site in range(length):
            for R in range(length):
                site = (R + relative_site) % length
                transform[4 * (R + 1) + site, relative_site + 1] = (
                    np.exp(1j * k * R) / 2
                )
        blocks.append(transform)
        sk = transform.conj().T @ s20 @ transform
        hk = transform.conj().T @ h20 @ transform
        assert np.linalg.eigvalsh(sk)[0] > 0.12
        energies.append(eigh(hk, sk, eigvals_only=True)[0])

    transform_all = np.concatenate(blocks, axis=1)
    np.testing.assert_allclose(
        transform_all.conj().T @ transform_all, np.eye(20), atol=1e-14
    )
    transformed_s = transform_all.conj().T @ s20 @ transform_all
    transformed_h = transform_all.conj().T @ h20 @ transform_all
    for m in range(length):
        for n in range(m + 1, length):
            rows = slice(5 * m, 5 * (m + 1))
            cols = slice(5 * n, 5 * (n + 1))
            assert np.linalg.norm(transformed_s[rows, cols]) < 1e-8
            assert np.linalg.norm(transformed_h[rows, cols]) < 1e-8
    assert energies[0] == pytest.approx(full_energy, abs=1e-11)
    assert min(energies[1:]) > energies[0] + 0.18
    assert full_energy == pytest.approx(-2.930809431151, abs=1e-10)

    # The original 12-frame recipe contains repeated symmetric frames.
    sym_lam, sym_shift = lf_mr.lf_translation_orbit(
        symmetric.lam, symmetric.shift
    )
    cs_lam, cs_shift = lf_mr.lf_translation_orbit(
        results22[0].lam, results22[0].shift
    )
    original_lam = np.concatenate((sym_lam, cs_lam, local_lam))
    original_shift = np.concatenate((sym_shift, cs_shift, local_shift))
    original_h = lf_mr.lf_frame_hamiltonian(
        tmat, g22, omega, original_lam, original_shift
    )
    original_s = lf_mr.lf_frame_overlap(original_lam, original_shift)
    original_energy, _, original_rank = lf_mr.lf_noci_lowest(
        original_h, original_s
    )
    assert original_rank == 20
    assert original_energy == pytest.approx(full_energy, abs=1e-11)

    # Project the actual 48-coordinate, 12-frame recipe before discarding
    # overlap-null directions. Its k=0 block has nominal size 12, rank 5.
    projection48 = np.zeros((48, 12))
    for family in range(3):
        for relative_site in range(length):
            for R in range(length):
                site = (R + relative_site) % length
                row = 4 * (family * 4 + R) + site
                projection48[row, family * 4 + relative_site] = 0.5
    s0 = projection48.T @ original_s.reshape(48, 48) @ projection48
    h0 = projection48.T @ original_h.reshape(48, 48) @ projection48
    # Matrix multiplication introduces roundoff-level asymmetry.
    s0 = (s0 + s0.T) / 2
    h0 = (h0 + h0.T) / 2
    projected_energy, _, projected_rank = lf_mr.lf_noci_lowest(
        h0.reshape(3, 4, 3, 4), s0.reshape(3, 4, 3, 4)
    )
    assert projected_rank == 5
    assert projected_energy == pytest.approx(original_energy, abs=1e-11)

    # Recenter the localized seed so reflection sends relative site r to -r.
    center = int(np.argmax(broken.density))
    seed_lam = np.roll(broken.lam, -center, axis=(0, 1))
    seed_shift = np.roll(broken.shift, -center)
    reflected = (-np.arange(length)) % length
    assert np.max(np.abs(
        seed_lam - seed_lam[np.ix_(reflected, reflected)]
    )) < 1e-8
    reflected_lam, reflected_shift = lf_mr.lf_translation_orbit(
        seed_lam, seed_shift
    )
    recentered_lam = np.concatenate((symmetric.lam[None], reflected_lam))
    recentered_shift = np.concatenate((symmetric.shift[None], reflected_shift))
    s20 = lf_mr.lf_frame_overlap(
        recentered_lam, recentered_shift
    ).reshape(20, 20)
    h20 = lf_mr.lf_frame_hamiltonian(
        tmat, g22, omega, recentered_lam, recentered_shift
    ).reshape(20, 20)
    parity = np.zeros((5, 5))
    parity[0, 0] = parity[1, 1] = parity[3, 3] = 1.0
    parity[2, 2] = parity[4, 2] = 1 / np.sqrt(2)
    parity[2, 4] = 1 / np.sqrt(2)
    parity[4, 4] = -1 / np.sqrt(2)
    even_odd = blocks[0].real @ parity
    sp = even_odd.T @ s20 @ even_odd
    hp = even_odd.T @ h20 @ even_odd
    assert np.linalg.norm(sp[:4, 4]) < 1e-8
    assert np.linalg.norm(hp[:4, 4]) < 1e-8
    even_energy = eigh(hp[:4, :4], sp[:4, :4], eigvals_only=True)[0]
    assert even_energy == pytest.approx(full_energy, abs=1e-10)
