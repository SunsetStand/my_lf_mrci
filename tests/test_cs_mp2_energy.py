"""Acceptance tests for the one-electron CS-MP2 energy correction."""

import numpy as np
import pytest

from src.cs_mp import (
    cs_mp2_mixed_energy,
    cs_mp2_mixed_intermediates,
    local_ep_coupling_mo,
)


def test_complex_matrix_elements_use_absolute_squares_without_prefactor():
    matrix_elements = np.array(
        [
            [[1.0 + 2.0j, 2.0 - 1.0j]],
            [[-1.0j, 0.5 + 0.5j]],
        ]
    )
    denominators = np.array(
        [
            [[-2.0, -4.0]],
            [[-1.0, -5.0]],
        ]
    )
    expected = sum(
        abs(matrix_elements[index]) ** 2 / denominators[index]
        for index in np.ndindex(matrix_elements.shape)
    )

    correction = cs_mp2_mixed_energy(matrix_elements, denominators)

    assert isinstance(correction, float)
    assert correction == pytest.approx(expected, abs=1e-14)
    assert correction < 0.0


def test_four_site_ring_recovers_analytic_cs_mp2_correction():
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

    correction = cs_mp2_mixed_energy(matrix_elements, denominators)

    assert correction == pytest.approx(-23.0 * coupling**2 / 90.0, abs=1e-14)


def test_zero_coupling_gives_zero_correction():
    matrix_elements = np.zeros((4, 1, 3))
    denominators = -np.ones((4, 1, 3))

    assert cs_mp2_mixed_energy(matrix_elements, denominators) == 0.0


def test_rejects_shape_mismatch_and_nonnegative_denominators():
    with pytest.raises(ValueError, match="same shape"):
        cs_mp2_mixed_energy(np.zeros((2, 1, 3)), -np.ones((2, 1, 2)))

    with pytest.raises(ValueError, match="negative"):
        cs_mp2_mixed_energy(np.zeros((1, 1, 2)), np.array([[[-1.0, 0.0]]]))
