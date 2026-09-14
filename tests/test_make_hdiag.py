"""Acceptance tests for the exact Hamiltonian diagonal."""

import numpy as np

from src.my_direct_ep import (
    contract_all,
    electron_ring_hopping,
    make_hdiag,
    make_shape,
    phonon_configs,
)


def _dense_diagonal(tmat, interaction, coupling, hpp, nsite, nelec, nmax):
    ci_shape = make_shape(nsite, nelec, nmax)
    dimension = int(np.prod(ci_shape))
    diagonal = np.empty(dimension)
    for address in range(dimension):
        basis_vector = np.zeros(dimension)
        basis_vector[address] = 1.0
        image = contract_all(
            tmat,
            interaction,
            coupling,
            hpp,
            basis_vector.reshape(ci_shape),
            nsite,
            nelec,
            nmax,
        ).reshape(-1)
        diagonal[address] = image[address]
    return diagonal


def test_one_site_matches_hand_calculation():
    result = make_hdiag(
        np.array([[0.2]]), 1.3, 0.3, np.array([[0.5]]), 1, (1, 1), 2
    )
    np.testing.assert_allclose(result, [1.7, 2.2, 2.7], atol=1e-12)


def test_matches_tiny_dense_hamiltonian_diagonal():
    nsite, nelec, nmax = 2, (1, 1), 2
    hopping = np.array([[0.2, -1.0], [-1.0, -0.1]])
    interaction = 1.3
    coupling = 0.37
    hpp = np.diag([0.5, 0.7])
    expected = _dense_diagonal(
        hopping, interaction, coupling, hpp, nsite, nelec, nmax
    )

    result = make_hdiag(
        hopping, interaction, coupling, hpp, nsite, nelec, nmax
    )

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_is_independent_of_electron_phonon_coupling():
    hopping = np.array([[0.2, -1.0], [-1.0, -0.1]])
    hpp = np.diag([0.5, 0.7])
    zero_coupling = make_hdiag(hopping, 1.3, 0.0, hpp, 2, (1, 1), 2)
    large_coupling = make_hdiag(hopping, 1.3, 9.0, hpp, 2, (1, 1), 2)
    np.testing.assert_allclose(large_coupling, zero_coupling, atol=1e-12)


def test_single_electron_is_independent_of_hubbard_interaction():
    hopping = electron_ring_hopping(4)
    hpp = 0.5 * np.eye(4)
    zero_interaction = make_hdiag(hopping, 0.0, 0.2, hpp, 4, (1, 0), 2)
    large_interaction = make_hdiag(hopping, 20.0, 0.2, hpp, 4, (1, 0), 2)
    np.testing.assert_allclose(large_interaction, zero_interaction, atol=1e-12)


def test_fig2b_shape_and_phonon_energies():
    nsite, nelec, nmax = 4, (1, 0), 2
    result = make_hdiag(
        electron_ring_hopping(nsite),
        0.0,
        0.2,
        0.5 * np.eye(nsite),
        nsite,
        nelec,
        nmax,
    )
    phonon_energies = 0.5 * phonon_configs(nsite, nmax).sum(axis=1)
    expected = np.broadcast_to(
        phonon_energies.reshape((1, 1) + (nmax + 1,) * nsite),
        make_shape(nsite, nelec, nmax),
    ).reshape(-1)

    assert result.shape == (324,)
    np.testing.assert_allclose(result, expected, atol=1e-12)
