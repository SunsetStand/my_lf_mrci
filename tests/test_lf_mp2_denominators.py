"""Acceptance tests for LF-MP2 zeroth-order energy denominators."""

import numpy as np
import pytest

from src.lf_mp import lf_mp2_denominators


def test_four_site_one_electron_denominators_match_hand_values():
    mo_energy = np.array([-2.0, 0.0, 0.0, 2.0])
    occupations = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [1, 0, 1, 0]])

    totals, pure, singles = lf_mp2_denominators(
        mo_energy, omega=0.5, occupations=occupations, nocc=1
    )

    np.testing.assert_array_equal(totals, [1, 2, 2])
    assert totals.dtype == np.int64
    np.testing.assert_allclose(pure, [-0.5, -1.0, -1.0], atol=1e-14)
    np.testing.assert_allclose(
        singles[:, 0, :],
        [
            [-2.5, -2.5, -4.5],
            [-3.0, -3.0, -5.0],
            [-3.0, -3.0, -5.0],
        ],
        atol=1e-14,
    )


def test_axes_are_configuration_occupied_virtual_for_multiple_occupied_mos():
    mo_energy = np.array([-3.0, -1.0, 0.5, 2.0])
    occupations = np.array([[1, 0], [1, 2]])
    omega = 0.4

    totals, pure, singles = lf_mp2_denominators(
        mo_energy, omega, occupations, nocc=2
    )

    assert totals.shape == (2,)
    assert pure.shape == (2,)
    assert singles.shape == (2, 2, 2)
    for config in range(2):
        for occupied in range(2):
            for virtual_offset in range(2):
                virtual_mo = 2 + virtual_offset
                assert singles[config, occupied, virtual_offset] == pytest.approx(
                    mo_energy[occupied]
                    - mo_energy[virtual_mo]
                    - totals[config] * omega
                )


def test_all_denominators_are_real_and_strictly_negative():
    _, pure, singles = lf_mp2_denominators(
        np.array([-1.0, -1.0, 0.2]),
        omega=0.3,
        occupations=np.array([[1, 0], [0, 4]]),
        nocc=2,
    )

    assert np.issubdtype(pure.dtype, np.floating)
    assert np.issubdtype(singles.dtype, np.floating)
    assert np.all(pure < 0.0)
    assert np.all(singles < 0.0)


@pytest.mark.parametrize(
    ("mo_energy", "omega", "occupations", "nocc", "exception"),
    [
        (np.zeros((2, 2)), 0.5, np.array([[1, 0]]), 1, ValueError),
        (np.array([-1.0, np.nan]), 0.5, np.array([[1, 0]]), 1, ValueError),
        (np.array([0.0, -1.0]), 0.5, np.array([[1, 0]]), 1, ValueError),
        (np.array([-1.0, 0.0]), 0.0, np.array([[1, 0]]), 1, ValueError),
        (np.array([-1.0, 0.0]), np.nan, np.array([[1, 0]]), 1, ValueError),
        (np.array([-1.0, 0.0]), 0.5, np.array([1, 0]), 1, ValueError),
        (np.array([-1.0, 0.0]), 0.5, np.array([[0, 0]]), 1, ValueError),
        (np.array([-1.0, 0.0]), 0.5, np.array([[0.0, 1.0]]), 1, ValueError),
        (np.array([-1.0, 0.0]), 0.5, np.array([[1, 0]]), 0, ValueError),
        (np.array([-1.0, 0.0]), 0.5, np.array([[1, 0]]), 2, ValueError),
        (np.array([-1.0, 0.0]), 0.5, np.array([[1, 0]]), 1.5, TypeError),
        (np.array([-1.0, 0.0]), 0.5, np.array([[1, 0]]), True, TypeError),
    ],
)
def test_rejects_invalid_energies_frequency_configurations_and_nocc(
    mo_energy, omega, occupations, nocc, exception
):
    with pytest.raises(exception):
        lf_mp2_denominators(mo_energy, omega, occupations, nocc)
