"""Acceptance tests for the local electron--phonon coupling in the MO basis."""

import numpy as np

from src.cs_mp import local_ep_coupling_mo


def test_complex_mo_coupling_reconstructs_each_local_site_operator():
    rng = np.random.default_rng(23)
    trial = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    mo_coeff = np.linalg.qr(trial)[0]
    coupling = 0.37

    coupling_mo = local_ep_coupling_mo(coupling, mo_coeff)

    assert coupling_mo.shape == (4, 4, 4)
    assert np.issubdtype(coupling_mo.dtype, np.complexfloating)
    for mode in range(4):
        np.testing.assert_allclose(
            coupling_mo[mode],
            coupling_mo[mode].conj().T,
            atol=1e-14,
        )
        reconstructed = mo_coeff @ coupling_mo[mode] @ mo_coeff.conj().T
        expected = np.zeros((4, 4), dtype=complex)
        expected[mode, mode] = coupling
        np.testing.assert_allclose(reconstructed, expected, atol=1e-14)


def test_uniform_occupied_orbital_obeys_cs_mp_sum_rule():
    hopping = np.array(
        [
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0, 0.0],
        ]
    )
    _, mo_coeff = np.linalg.eigh(hopping)
    coupling = 0.6

    coupling_mo = local_ep_coupling_mo(coupling, mo_coeff)

    occupied = 0
    np.testing.assert_allclose(coupling_mo[:, occupied, occupied], coupling / 4)
    for virtual in range(1, 4):
        np.testing.assert_allclose(
            np.sum(np.abs(coupling_mo[:, virtual, occupied]) ** 2),
            coupling**2 / 4,
            atol=1e-14,
        )

