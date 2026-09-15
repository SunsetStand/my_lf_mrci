"""Acceptance tests for the centered electron-phonon contraction."""

import numpy as np

from src.my_direct_ep import (
    boson_operators,
    contract_ep_centered,
    contract_ep_paper,
    make_shape,
)


def _uniform_force(g, psi, nsite, nelec, nmax):
    """Apply g * (Ne/L) * sum_i X_i as a small dense reference."""
    identity = np.eye(nmax + 1)
    b, bdag, _ = boson_operators(nmax)
    displacement = b + bdag
    phonon_dimension = (nmax + 1) ** nsite
    total_displacement = np.zeros((phonon_dimension, phonon_dimension))
    for site in range(nsite):
        factors = [identity] * nsite
        factors[site] = displacement
        site_displacement = factors[0]
        for factor in factors[1:]:
            site_displacement = np.kron(site_displacement, factor)
        total_displacement += site_displacement

    electron_dimension = psi.shape[0] * psi.shape[1]
    flat = psi.reshape(electron_dimension, phonon_dimension)
    average_occupation = sum(nelec) / nsite
    return (g * average_occupation * flat @ total_displacement.T).reshape(psi.shape)


def test_zero_coupling_and_zero_phonon_cutoff_give_zero():
    rng = np.random.default_rng(31)
    psi = rng.normal(size=make_shape(3, (1, 1), 2))
    np.testing.assert_array_equal(
        contract_ep_centered(0.0, psi, 3, (1, 1), 2), np.zeros_like(psi)
    )

    psi_nmax_zero = rng.normal(size=make_shape(3, (1, 1), 0))
    np.testing.assert_array_equal(
        contract_ep_centered(0.7, psi_nmax_zero, 3, (1, 1), 0),
        np.zeros_like(psi_nmax_zero),
    )


def test_two_site_matrix_matches_centered_kronecker_reference():
    nsite, nelec, nmax, coupling = 2, (1, 1), 1, 0.41
    shape = make_shape(nsite, nelec, nmax)
    dimension = int(np.prod(shape))
    contracted = np.empty((dimension, dimension))
    for column in range(dimension):
        basis = np.zeros(dimension)
        basis[column] = 1.0
        contracted[:, column] = contract_ep_centered(
            coupling, basis.reshape(shape), nsite, nelec, nmax
        ).reshape(-1)

    annihilation = np.array([[0.0, 1.0], [0.0, 0.0]])
    displacement = annihilation + annihilation.T
    identity_phonon = np.eye(2)
    centered_site_zero = np.diag([1.0, 0.0, 0.0, -1.0])
    centered_site_one = np.diag([-1.0, 0.0, 0.0, 1.0])
    reference = coupling * (
        np.kron(centered_site_zero, np.kron(displacement, identity_phonon))
        + np.kron(centered_site_one, np.kron(identity_phonon, displacement))
    )

    np.testing.assert_allclose(contracted, reference, atol=1e-12)


def test_paper_minus_centered_is_uniform_force_for_complex_input():
    rng = np.random.default_rng(32)
    nsite, nelec, nmax, coupling = 3, (1, 1), 2, 0.37
    shape = make_shape(nsite, nelec, nmax)
    psi = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    difference = contract_ep_paper(
        coupling, psi, nsite, nelec, nmax
    ) - contract_ep_centered(coupling, psi, nsite, nelec, nmax)
    reference = _uniform_force(coupling, psi, nsite, nelec, nmax)

    np.testing.assert_allclose(difference, reference, atol=1e-12)


def test_centered_coupling_annihilates_singly_occupied_determinant():
    shape = make_shape(4, (2, 2), 2)
    psi = np.zeros(shape)
    # Alpha string 0011 and beta string 1100: exactly one electron per site.
    psi[(0, 5) + (0,) * 4] = 1.0

    result = contract_ep_centered(0.41, psi, 4, (2, 2), 2)

    np.testing.assert_array_equal(result, np.zeros_like(result))
