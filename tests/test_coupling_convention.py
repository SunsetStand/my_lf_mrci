"""Acceptance tests for explicit paper/centered Hamiltonian dispatch."""

import numpy as np
import pytest

from src.my_direct_ep import (
    contract_1e,
    contract_2e_hubbard,
    contract_all,
    contract_ep_centered,
    contract_ep_paper,
    contract_pp,
    electron_ring_hopping,
    kernel,
    make_shape,
)


def _parameters():
    nsite, nelec, nmax = 3, (1, 1), 2
    return (
        electron_ring_hopping(nsite),
        1.3,
        0.27,
        0.5 * np.eye(nsite),
        nsite,
        nelec,
        nmax,
    )


def test_contract_all_default_and_explicit_paper_are_backward_compatible():
    hopping, interaction, coupling, hpp, nsite, nelec, nmax = _parameters()
    rng = np.random.default_rng(51)
    psi = rng.normal(size=make_shape(nsite, nelec, nmax))
    expected = (
        contract_1e(hopping, psi, nsite, nelec, nmax)
        + contract_2e_hubbard(interaction, psi, nsite, nelec, nmax)
        + contract_pp(hpp, psi, nsite, nelec, nmax)
        + contract_ep_paper(coupling, psi, nsite, nelec, nmax)
    )

    default = contract_all(
        hopping, interaction, coupling, hpp, psi, nsite, nelec, nmax
    )
    explicit = contract_all(
        hopping,
        interaction,
        coupling,
        hpp,
        psi,
        nsite,
        nelec,
        nmax,
        coupling_convention="paper",
    )

    np.testing.assert_allclose(default, expected, atol=1e-12)
    np.testing.assert_allclose(explicit, expected, atol=1e-12)


def test_contract_all_centered_selects_centered_coupling():
    hopping, interaction, coupling, hpp, nsite, nelec, nmax = _parameters()
    rng = np.random.default_rng(52)
    psi = rng.normal(size=make_shape(nsite, nelec, nmax))
    expected = (
        contract_1e(hopping, psi, nsite, nelec, nmax)
        + contract_2e_hubbard(interaction, psi, nsite, nelec, nmax)
        + contract_pp(hpp, psi, nsite, nelec, nmax)
        + contract_ep_centered(coupling, psi, nsite, nelec, nmax)
    )

    result = contract_all(
        hopping,
        interaction,
        coupling,
        hpp,
        psi,
        nsite,
        nelec,
        nmax,
        coupling_convention="centered",
    )

    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("invalid", ["centre", "uncentered", "", None])
def test_contract_all_rejects_unknown_convention(invalid):
    hopping, interaction, coupling, hpp, nsite, nelec, nmax = _parameters()
    psi = np.zeros(make_shape(nsite, nelec, nmax))

    with pytest.raises(ValueError, match="coupling_convention"):
        contract_all(
            hopping,
            interaction,
            coupling,
            hpp,
            psi,
            nsite,
            nelec,
            nmax,
            coupling_convention=invalid,
        )


def test_kernel_passes_centered_convention_to_hamiltonian():
    nsite, nelec, nmax = 2, (1, 1), 1
    hopping = np.array([[0.0, -1.0], [-1.0, 0.0]])
    interaction, coupling, hpp = 1.3, 0.37, 0.5 * np.eye(nsite)
    shape = make_shape(nsite, nelec, nmax)
    dimension = int(np.prod(shape))
    dense = np.empty((dimension, dimension))
    for column in range(dimension):
        basis = np.zeros(dimension)
        basis[column] = 1.0
        dense[:, column] = (
            contract_1e(hopping, basis.reshape(shape), nsite, nelec, nmax)
            + contract_2e_hubbard(
                interaction, basis.reshape(shape), nsite, nelec, nmax
            )
            + contract_pp(hpp, basis.reshape(shape), nsite, nelec, nmax)
            + contract_ep_centered(
                coupling, basis.reshape(shape), nsite, nelec, nmax
            )
        ).reshape(-1)
    expected = np.linalg.eigvalsh(dense)[0]

    energy, _, residual = kernel(
        hopping,
        interaction,
        coupling,
        hpp,
        nsite,
        nelec,
        nmax,
        tol_residual=1e-10,
        coupling_convention="centered",
    )

    np.testing.assert_allclose(energy, expected, atol=1e-11)
    assert residual < 1e-10
