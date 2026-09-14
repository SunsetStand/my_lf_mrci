"""Acceptance tests for the complete matrix-free Hamiltonian action."""

import numpy as np
import pytest

from src.my_direct_ep import (
    contract_1e,
    contract_2e_hubbard,
    contract_all,
    contract_ep_paper,
    contract_pp,
    electron_ring_hopping,
    make_shape,
)


def test_matches_sum_of_four_terms_without_mutating_input():
    rng = np.random.default_rng(44)
    nsite, nelec, nmax = 3, (1, 1), 2
    psi = rng.normal(size=make_shape(nsite, nelec, nmax))
    before = psi.copy()
    hopping = electron_ring_hopping(nsite)
    interaction = 1.3
    coupling = 0.27
    hpp = 0.5 * np.eye(nsite)
    expected = (
        contract_1e(hopping, psi, nsite, nelec, nmax)
        + contract_2e_hubbard(interaction, psi, nsite, nelec, nmax)
        + contract_ep_paper(coupling, psi, nsite, nelec, nmax)
        + contract_pp(hpp, psi, nsite, nelec, nmax)
    )

    result = contract_all(
        hopping, interaction, coupling, hpp, psi, nsite, nelec, nmax
    )

    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_array_equal(psi, before)


def test_zero_hamiltonian_returns_zero():
    nsite, nelec, nmax = 3, (1, 1), 1
    psi = np.ones(make_shape(nsite, nelec, nmax))

    result = contract_all(
        np.zeros((nsite, nsite)),
        0.0,
        0.0,
        np.zeros((nsite, nsite)),
        psi,
        nsite,
        nelec,
        nmax,
    )

    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_four_site_k_zero_phonon_vacuum_has_energy_minus_two():
    nsite, nelec, nmax = 4, (1, 0), 1
    psi = np.zeros(make_shape(nsite, nelec, nmax))
    psi[(slice(None), 0) + (0,) * nsite] = 0.5

    result = contract_all(
        electron_ring_hopping(nsite, -1.0),
        0.0,
        0.0,
        0.5 * np.eye(nsite),
        psi,
        nsite,
        nelec,
        nmax,
    )

    np.testing.assert_allclose(result, -2.0 * psi, atol=1e-12)


def test_is_linear():
    rng = np.random.default_rng(91)
    nsite, nelec, nmax = 3, (1, 0), 1
    shape = make_shape(nsite, nelec, nmax)
    left = rng.normal(size=shape)
    right = rng.normal(size=shape)
    a, b = 0.31, -1.2
    arguments = (
        electron_ring_hopping(nsite),
        0.7,
        0.2,
        0.4 * np.eye(nsite),
    )

    combined = contract_all(
        *arguments, a * left + b * right, nsite, nelec, nmax
    )
    separate = a * contract_all(
        *arguments, left, nsite, nelec, nmax
    ) + b * contract_all(*arguments, right, nsite, nelec, nmax)

    np.testing.assert_allclose(combined, separate, atol=1e-12)


def test_rejects_incompatible_wavefunction_shape():
    with pytest.raises(ValueError):
        contract_all(
            electron_ring_hopping(4),
            0.0,
            0.0,
            0.5 * np.eye(4),
            np.zeros((4, 1, 16)),
            4,
            (1, 0),
            1,
        )
