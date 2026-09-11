"""Acceptance tests for the electronic one-body contraction."""

import numpy as np
import pytest

from src.my_direct_ep import contract_1e, electron_ring_hopping, make_shape


@pytest.mark.parametrize("nelec", [(1, 0), (0, 1), (1, 1)])
def test_against_dense_reference(nelec):
    rng = np.random.default_rng(42)
    nsite, nmax = 4, 1
    phonon_dimension = (nmax + 1) ** nsite
    hopping = electron_ring_hopping(nsite)
    shape = make_shape(nsite, nelec, nmax)
    psi = rng.normal(size=shape)
    before = psi.copy()

    result = contract_1e(hopping, psi, nsite, nelec, nmax)
    electronic_hamiltonian = hopping
    if nelec == (1, 1):
        electronic_hamiltonian = (
            np.kron(hopping, np.eye(nsite))
            + np.kron(np.eye(nsite), hopping)
        )
    reference = (
        np.kron(electronic_hamiltonian, np.eye(phonon_dimension))
        @ psi.ravel()
    )

    assert result.shape == shape
    assert result.dtype == np.float64
    np.testing.assert_array_equal(psi, before)
    np.testing.assert_allclose(result.ravel(), reference, atol=1e-12)


def test_identity_counts_electrons():
    rng = np.random.default_rng(7)
    psi = rng.normal(size=make_shape(4, (2, 1), 1))
    result = contract_1e(np.eye(4), psi, 4, (2, 1), 1)
    np.testing.assert_allclose(result, 3 * psi, atol=1e-12)


def test_two_electron_fermionic_sign():
    psi = np.zeros(make_shape(4, (2, 0), 0))
    psi[1, 0, 0, 0, 0, 0] = 1.0  # 0101
    hopping = np.zeros((4, 4))
    hopping[3, 0] = hopping[0, 3] = -1.0
    expected = np.zeros_like(psi)
    expected[5, 0, 0, 0, 0, 0] = 1.0  # 1100
    np.testing.assert_allclose(
        contract_1e(hopping, psi, 4, (2, 0), 0), expected, atol=1e-12
    )


def test_electronic_vacuum_is_zero():
    psi = np.ones(make_shape(4, (0, 0), 1))
    result = contract_1e(electron_ring_hopping(4), psi, 4, (0, 0), 1)
    np.testing.assert_allclose(result, 0.0)


@pytest.mark.parametrize(
    "hopping, psi",
    [
        (np.zeros((3, 3)), np.zeros(make_shape(4, (1, 0), 1))),
        (np.zeros((4, 4)), np.zeros((4, 1, 16))),
        ([[0.0] * 4] * 4, np.zeros(make_shape(4, (1, 0), 1))),
        (np.zeros((4, 4)), [0.0]),
    ],
)
def test_rejects_incompatible_inputs(hopping, psi):
    with pytest.raises(ValueError):
        contract_1e(hopping, psi, 4, (1, 0), 1)
