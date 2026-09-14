"""Acceptance tests for the uncentered electron-phonon contraction."""

import numpy as np
import pytest

from src.my_direct_ep import (
    boson_operators,
    contract_ep_paper,
    make_electron_basis,
    make_shape,
)


def test_single_site_matches_local_boson_matrix():
    nmax, coupling = 3, 0.7
    psi = np.array([[[1.0, -0.5, 0.25, 0.1]]])
    b, bdag, _ = boson_operators(nmax)
    expected = coupling * (b + bdag) @ psi[0, 0]

    result = contract_ep_paper(coupling, psi, 1, (1, 0), nmax)

    np.testing.assert_allclose(result[0, 0], expected, atol=1e-12)


def test_two_site_matches_full_kronecker_reference():
    nsite, nmax, coupling = 2, 2, 0.43
    nelec = (1, 0)
    rng = np.random.default_rng(27)
    psi = rng.normal(size=make_shape(nsite, nelec, nmax))
    before = psi.copy()
    strings, _ = make_electron_basis(nsite, nelec[0])
    b, bdag, _ = boson_operators(nmax)
    displacement = b + bdag
    identity = np.eye(nmax + 1)
    site_operators = [
        np.kron(displacement, identity),
        np.kron(identity, displacement),
    ]
    expected = np.zeros_like(psi)
    for address, string in enumerate(strings):
        full_operator = sum(
            bool(string & (1 << site)) * site_operators[site]
            for site in range(nsite)
        )
        expected[address, 0] = (
            coupling * full_operator @ psi[address, 0].reshape(-1)
        ).reshape((nmax + 1,) * nsite)

    result = contract_ep_paper(coupling, psi, nsite, nelec, nmax)

    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_array_equal(psi, before)


def test_locality_on_every_phonon_axis():
    nsite, nmax, coupling = 4, 1, 1.25
    shape = make_shape(nsite, (1, 0), nmax)
    for electron_site in range(nsite):
        psi = np.zeros(shape)
        psi[(electron_site, 0) + (0,) * nsite] = 1.0
        expected = np.zeros_like(psi)
        phonons = [0] * nsite
        phonons[electron_site] = 1
        expected[(electron_site, 0, *phonons)] = coupling

        result = contract_ep_paper(coupling, psi, nsite, (1, 0), nmax)

        np.testing.assert_allclose(result, expected, atol=1e-12)


def test_double_occupation_gives_factor_two():
    psi = np.zeros(make_shape(1, (1, 1), 2))
    psi[0, 0, 0] = 1.0
    expected = np.zeros_like(psi)
    expected[0, 0, 1] = 0.8

    result = contract_ep_paper(0.4, psi, 1, (1, 1), 2)

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_complex_wavefunctions_obey_hermitian_identity():
    rng = np.random.default_rng(20260913)
    shape = make_shape(3, (1, 1), 2)
    left = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    right = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    h_left = contract_ep_paper(0.31, left, 3, (1, 1), 2)
    h_right = contract_ep_paper(0.31, right, 3, (1, 1), 2)

    assert h_left.dtype == np.complex128
    np.testing.assert_allclose(
        np.vdot(left, h_right), np.vdot(h_left, right), atol=1e-12
    )


@pytest.mark.parametrize(
    "invalid", [True, np.bool_(False), "0.5", 1j, np.nan, np.inf, -np.inf]
)
def test_rejects_non_real_or_non_finite_coupling(invalid):
    psi = np.zeros(make_shape(1, (1, 0), 1))
    with pytest.raises(ValueError):
        contract_ep_paper(invalid, psi, 1, (1, 0), 1)


def test_rejects_incompatible_wavefunction_shape():
    with pytest.raises(ValueError):
        contract_ep_paper(0.5, np.zeros((2, 1, 4)), 2, (1, 0), 1)
