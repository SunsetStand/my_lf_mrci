"""Acceptance tests for the matrix-free Davidson kernel."""

import numpy as np

from src.my_direct_ep import contract_all, electron_ring_hopping, kernel, make_shape
from tests.test_dense_reference import _build_dense_reference


def test_four_site_zero_coupling_ground_state():
    energy, psi, residual = kernel(
        electron_ring_hopping(4),
        0.0,
        0.0,
        0.5 * np.eye(4),
        4,
        (1, 0),
        0,
    )

    assert isinstance(energy, float)
    assert isinstance(residual, float)
    assert psi.shape == make_shape(4, (1, 0), 0)
    np.testing.assert_allclose(energy, -2.0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-12)
    assert residual < 1e-10


def test_single_site_matches_analytic_ground_energy():
    coupling, omega = 0.3, 0.5
    expected = (omega - np.sqrt(omega**2 + 4.0 * coupling**2)) / 2.0

    energy, psi, residual = kernel(
        np.zeros((1, 1)),
        0.0,
        coupling,
        np.array([[omega]]),
        1,
        (1, 0),
        1,
    )

    np.testing.assert_allclose(energy, expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-12)
    assert residual < 1e-10


def test_nonzero_coupling_matches_tiny_dense_ground_state_and_residual():
    nsite, nelec, nmax = 2, (1, 0), 2
    hopping = np.array([[0.0, -1.0], [-1.0, 0.0]])
    interaction, coupling = 0.0, 0.37
    hpp = 0.5 * np.eye(nsite)
    dense = _build_dense_reference(
        hopping, interaction, coupling, hpp, nsite, nelec, nmax
    )
    expected_energy = np.linalg.eigvalsh(dense)[0]

    energy, psi, residual = kernel(
        hopping, interaction, coupling, hpp, nsite, nelec, nmax
    )
    hpsi = contract_all(
        hopping, interaction, coupling, hpp, psi, nsite, nelec, nmax
    )
    direct_residual = np.linalg.norm(hpsi.reshape(-1) - energy * psi.reshape(-1))

    np.testing.assert_allclose(energy, expected_energy, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(psi), 1.0, atol=1e-12)
    np.testing.assert_allclose(residual, direct_residual, atol=1e-14)
    assert residual < 1e-6
