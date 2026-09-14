"""Tiny dense references used only to validate the matrix-free Hamiltonian."""

import numpy as np

from src.my_direct_ep import contract_all, electron_ring_hopping, make_shape


def _build_dense_reference(tmat, interaction, coupling, hpp, nsite, nelec, nmax):
    """Build H column by column; never use this helper for production scans."""
    ci_shape = make_shape(nsite, nelec, nmax)
    dimension = int(np.prod(ci_shape))
    hamiltonian = np.empty((dimension, dimension), dtype=np.float64)
    for column in range(dimension):
        basis_vector = np.zeros(dimension, dtype=np.float64)
        basis_vector[column] = 1.0
        hamiltonian[:, column] = contract_all(
            tmat,
            interaction,
            coupling,
            hpp,
            basis_vector.reshape(ci_shape),
            nsite,
            nelec,
            nmax,
        ).reshape(-1)
    return hamiltonian


def test_one_site_matrix_matches_hand_calculation():
    hamiltonian = _build_dense_reference(
        np.zeros((1, 1)), 0.0, 0.3, np.array([[0.5]]), 1, (1, 0), 1
    )
    expected = np.array([[0.0, 0.3], [0.3, 0.5]])
    np.testing.assert_allclose(hamiltonian, expected, atol=1e-12)


def test_nonzero_coupling_hamiltonian_is_symmetric():
    hopping = np.array([[0.0, -1.0], [-1.0, 0.0]])
    hamiltonian = _build_dense_reference(
        hopping, 0.0, 0.37, 0.5 * np.eye(2), 2, (1, 0), 2
    )
    np.testing.assert_allclose(hamiltonian, hamiltonian.T, atol=1e-12)


def test_four_site_zero_coupling_spectrum():
    hamiltonian = _build_dense_reference(
        electron_ring_hopping(4, -1.0),
        0.0,
        0.0,
        0.5 * np.eye(4),
        4,
        (1, 0),
        0,
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(hamiltonian), [-2.0, 0.0, 0.0, 2.0], atol=1e-12
    )
