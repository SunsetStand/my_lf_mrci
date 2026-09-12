"""Acceptance tests for local diagonal phonon-energy contraction."""

import numpy as np
import pytest

from src.my_direct_ep import contract_pp, make_shape, phonon_configs


def test_two_site_hand_calculated_table():
    nsite, nmax = 2, 2
    hpp = np.diag([0.5, 1.0])
    psi = np.ones(make_shape(nsite, (1, 0), nmax))
    result = contract_pp(hpp, psi, nsite, (1, 0), nmax)
    expected = np.array(
        [[0.0, 1.0, 2.0], [0.5, 1.5, 2.5], [1.0, 2.0, 3.0]]
    )
    for electronic_address in range(2):
        np.testing.assert_allclose(
            result[electronic_address, 0], expected, atol=1e-12
        )


def test_fig2b_frequency_against_configuration_reference():
    rng = np.random.default_rng(42)
    nsite, nmax, nelec, omega = 4, 2, (1, 0), 0.5
    psi = rng.normal(size=make_shape(nsite, nelec, nmax))
    before = psi.copy()
    result = contract_pp(omega * np.eye(nsite), psi, nsite, nelec, nmax)
    energies = omega * phonon_configs(nsite, nmax).sum(axis=1)
    expected = energies.reshape((1, 1) + (nmax + 1,) * nsite) * psi
    assert result.shape == psi.shape
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_array_equal(psi, before)


def test_independent_of_electronic_state():
    rng = np.random.default_rng(12)
    nsite, nmax, nelec = 4, 1, (2, 1)
    frequencies = np.array([0.2, 0.4, 0.6, 0.8])
    psi = rng.normal(size=make_shape(nsite, nelec, nmax))
    energies = phonon_configs(nsite, nmax) @ frequencies
    expected = energies.reshape((1, 1, 2, 2, 2, 2)) * psi
    result = contract_pp(np.diag(frequencies), psi, nsite, nelec, nmax)
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_phonon_vacuum_has_zero_energy():
    psi = np.ones(make_shape(4, (1, 0), 0))
    result = contract_pp(0.5 * np.eye(4), psi, 4, (1, 0), 0)
    np.testing.assert_allclose(result, 0.0)


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros((3, 3)),
        np.eye(4) + np.diag(np.ones(3), 1),
        np.diag([0.5, np.nan, 0.5, 0.5]),
        np.eye(4, dtype=complex),
        np.eye(4).tolist(),
        "hpp",
    ],
)
def test_rejects_invalid_hpp(invalid):
    psi = np.zeros(make_shape(4, (1, 0), 1))
    with pytest.raises(ValueError):
        contract_pp(invalid, psi, 4, (1, 0), 1)


def test_rejects_incompatible_wavefunction_shape():
    with pytest.raises(ValueError):
        contract_pp(0.5 * np.eye(4), np.zeros((4, 1, 16)), 4, (1, 0), 1)
