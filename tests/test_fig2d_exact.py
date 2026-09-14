"""Independent exact-diagonalization checks for the Fig. 2d sector."""

import numpy as np
from pyscf import fci

from src.my_direct_ep import (
    contract_all,
    electron_ring_hopping,
    kernel,
    make_hdiag,
    make_shape,
)


def _manual_two_site_hamiltonian(
    hopping: np.ndarray,
    interaction: float,
    coupling: float,
    omega: float,
) -> np.ndarray:
    """Build the L=2, (1,1), Nmax=1 matrix without model contractions."""
    identity_electron_spin = np.eye(2)
    h_electron = (
        np.kron(hopping, identity_electron_spin)
        + np.kron(identity_electron_spin, hopping)
        + np.diag([interaction, 0.0, 0.0, interaction])
    )

    annihilation = np.array([[0.0, 1.0], [0.0, 0.0]])
    displacement = annihilation + annihilation.T
    number = annihilation.T @ annihilation
    identity_phonon = np.eye(2)
    x_site = (
        np.kron(displacement, identity_phonon),
        np.kron(identity_phonon, displacement),
    )
    h_phonon = omega * (
        np.kron(number, identity_phonon)
        + np.kron(identity_phonon, number)
    )

    # Electronic order: (alpha site, beta site) = 00, 01, 10, 11.
    occupation_site = (
        np.diag([2.0, 1.0, 1.0, 0.0]),
        np.diag([0.0, 1.0, 1.0, 2.0]),
    )
    identity_electron = np.eye(4)
    identity_all_phonons = np.eye(4)
    return (
        np.kron(h_electron, identity_all_phonons)
        + np.kron(identity_electron, h_phonon)
        + coupling
        * sum(
            np.kron(occupation_site[site], x_site[site]) for site in range(2)
        )
    )


def _contracted_two_site_hamiltonian(
    hopping: np.ndarray,
    interaction: float,
    coupling: float,
    omega: float,
) -> np.ndarray:
    shape = make_shape(2, (1, 1), 1)
    dimension = int(np.prod(shape))
    result = np.empty((dimension, dimension))
    for column in range(dimension):
        basis = np.zeros(dimension)
        basis[column] = 1.0
        result[:, column] = contract_all(
            hopping,
            interaction,
            coupling,
            omega * np.eye(2),
            basis.reshape(shape),
            2,
            (1, 1),
            1,
        ).reshape(-1)
    return result


def test_zero_coupling_fig2d_sector_matches_pyscf_electronic_fci():
    nsite = 4
    nelec = (2, 2)
    interaction = 4.0
    hopping = electron_ring_hopping(nsite, -1.0)
    eri = np.zeros((nsite, nsite, nsite, nsite))
    sites = np.arange(nsite)
    eri[sites, sites, sites, sites] = interaction
    reference_energy, _ = fci.direct_spin1.kernel(
        hopping, eri, nsite, nelec, tol=1e-13
    )

    energy, _, residual = kernel(
        hopping,
        interaction,
        0.0,
        0.5 * np.eye(nsite),
        nsite,
        nelec,
        0,
        tol_residual=1e-10,
    )

    np.testing.assert_allclose(reference_energy, -2.102748483462, atol=1e-12)
    np.testing.assert_allclose(energy, reference_energy, atol=1e-12)
    assert residual < 1e-10


def test_nonzero_coupling_matches_independent_kronecker_matrix():
    hopping = np.array([[0.0, -1.0], [-1.0, 0.0]])
    interaction, coupling, omega = 1.3, 0.37, 0.5
    reference = _manual_two_site_hamiltonian(
        hopping, interaction, coupling, omega
    )
    contracted = _contracted_two_site_hamiltonian(
        hopping, interaction, coupling, omega
    )

    np.testing.assert_allclose(contracted, reference, atol=1e-12)
    np.testing.assert_allclose(contracted, contracted.T, atol=1e-12)
    np.testing.assert_allclose(
        make_hdiag(
            hopping,
            interaction,
            coupling,
            omega * np.eye(2),
            2,
            (1, 1),
            1,
        ),
        np.diag(reference),
        atol=1e-12,
    )


def test_nonzero_coupling_davidson_matches_independent_dense_ground_state():
    hopping = np.array([[0.0, -1.0], [-1.0, 0.0]])
    interaction, coupling, omega = 1.3, 0.37, 0.5
    reference = _manual_two_site_hamiltonian(
        hopping, interaction, coupling, omega
    )
    expected_energy = np.linalg.eigvalsh(reference)[0]

    energy, _, residual = kernel(
        hopping,
        interaction,
        coupling,
        omega * np.eye(2),
        2,
        (1, 1),
        1,
        tol_residual=1e-10,
    )

    np.testing.assert_allclose(energy, expected_energy, atol=1e-11)
    assert residual < 1e-10
