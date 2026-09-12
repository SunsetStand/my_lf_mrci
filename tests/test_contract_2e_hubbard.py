"""Acceptance tests for the on-site Hubbard contraction."""

import numpy as np
import pytest

from src.my_direct_ep import contract_2e_hubbard, make_shape


@pytest.mark.parametrize("nelec", [(1, 0), (0, 1)])
def test_single_electron_has_no_double_occupation(nelec):
    rng = np.random.default_rng(42)
    psi = rng.normal(size=make_shape(4, nelec, 1))
    before = psi.copy()
    result = contract_2e_hubbard(3.0, psi, 4, nelec, 1)
    np.testing.assert_allclose(result, 0.0)
    np.testing.assert_array_equal(psi, before)


def test_one_alpha_one_beta_diagonal_pattern():
    interaction = 2.5
    psi = np.ones(make_shape(4, (1, 1), 1))
    result = contract_2e_hubbard(interaction, psi, 4, (1, 1), 1)
    electronic_factor = interaction * np.eye(4)
    expected = np.broadcast_to(
        electronic_factor[:, :, None, None, None, None], psi.shape
    )
    np.testing.assert_allclose(result, expected)


def test_multiple_electrons_against_bit_count_reference():
    rng = np.random.default_rng(17)
    interaction = 2.5
    psi = rng.normal(size=make_shape(4, (2, 2), 1))
    before = psi.copy()
    result = contract_2e_hubbard(interaction, psi, 4, (2, 2), 1)
    strings = np.array([3, 5, 6, 9, 10, 12])
    factor = np.empty((6, 6))
    for ia, string_a in enumerate(strings):
        for ib, string_b in enumerate(strings):
            factor[ia, ib] = interaction * int(string_a & string_b).bit_count()
    expected = factor[:, :, None, None, None, None] * psi
    np.testing.assert_allclose(result, expected, atol=1e-12)
    np.testing.assert_array_equal(psi, before)


def test_filled_sector_has_one_double_occupation_per_site():
    rng = np.random.default_rng(8)
    interaction = 2.5
    psi = rng.normal(size=make_shape(4, (4, 4), 1))
    result = contract_2e_hubbard(interaction, psi, 4, (4, 4), 1)
    np.testing.assert_allclose(result, 4 * interaction * psi)


@pytest.mark.parametrize("invalid", [np.inf, -np.inf, np.nan, "2.0"])
def test_rejects_invalid_interaction(invalid):
    psi = np.zeros(make_shape(4, (1, 1), 1))
    with pytest.raises(ValueError):
        contract_2e_hubbard(invalid, psi, 4, (1, 1), 1)


def test_rejects_incompatible_wavefunction_shape():
    with pytest.raises(ValueError):
        contract_2e_hubbard(2.0, np.zeros((4, 4, 16)), 4, (1, 1), 1)
