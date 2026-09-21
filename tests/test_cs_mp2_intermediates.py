"""Acceptance tests for the CS-MP2 mixed electron--phonon intermediates."""

import numpy as np

from src.cs_mp import cs_mp2_mixed_intermediates, local_ep_coupling_mo


def test_intermediate_axes_use_mode_occupied_virtual_order():
    mo_energy = np.array([-3.0, -1.0, 0.5, 2.0])
    coupling_mo = np.empty((2, 4, 4), dtype=complex)
    for mode in range(2):
        for p in range(4):
            for q in range(4):
                coupling_mo[mode, p, q] = 100 * mode + 10 * p + q + 0.25j

    matrix_elements, denominators = cs_mp2_mixed_intermediates(
        mo_energy,
        coupling_mo,
        omega=0.5,
        nocc=2,
    )

    assert matrix_elements.shape == (2, 2, 2)
    assert denominators.shape == (2, 2, 2)
    assert np.issubdtype(matrix_elements.dtype, np.complexfloating)
    assert np.issubdtype(denominators.dtype, np.floating)
    for mode in range(2):
        for occupied in range(2):
            for virtual_offset in range(2):
                virtual_mo = 2 + virtual_offset
                assert (
                    matrix_elements[mode, occupied, virtual_offset]
                    == coupling_mo[mode, virtual_mo, occupied]
                )
                assert denominators[mode, occupied, virtual_offset] == (
                    mo_energy[occupied] - mo_energy[virtual_mo] - 0.5
                )


def test_four_site_ring_recovers_hand_derived_denominators_and_strengths():
    hopping = np.array(
        [
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0, 0.0],
            [0.0, -1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0, 0.0],
        ]
    )
    coupling = 0.6
    omega = 0.5
    shift = np.full(4, -coupling / (4 * omega))
    fock = hopping + np.diag(2 * coupling * shift)
    mo_energy, mo_coeff = np.linalg.eigh(fock)
    coupling_mo = local_ep_coupling_mo(coupling, mo_coeff)

    matrix_elements, denominators = cs_mp2_mixed_intermediates(
        mo_energy,
        coupling_mo,
        omega,
        nocc=1,
    )

    assert matrix_elements.shape == (4, 1, 3)
    assert denominators.shape == (4, 1, 3)
    np.testing.assert_allclose(
        denominators[:, 0, :],
        np.tile([-2.5, -2.5, -4.5], (4, 1)),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.sum(np.abs(matrix_elements[:, 0, :]) ** 2, axis=0),
        coupling**2 / 4,
        atol=1e-14,
    )
    assert np.all(np.abs(matrix_elements) ** 2 / denominators <= 0.0)
