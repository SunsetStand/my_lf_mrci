"""Acceptance tests for the zero-phonon Franck--Condon factor."""

import numpy as np
import pytest

from src.lf_mp import lf_franck_condon


def test_zero_rectangular_displacements_give_unit_overlaps():
    lam = np.zeros((2, 3))
    original = lam.copy()

    overlap = lf_franck_condon(lam)

    assert overlap.shape == (3, 3)
    assert overlap.dtype == np.float64
    np.testing.assert_array_equal(overlap, np.ones((3, 3)))
    np.testing.assert_array_equal(lam, original)


def test_local_displacements_have_expected_off_diagonal_overlap():
    displacement = 0.7
    lam = displacement * np.eye(4)
    expected = np.full((4, 4), np.exp(-(displacement**2)))
    np.fill_diagonal(expected, 1.0)

    overlap = lf_franck_condon(lam)

    np.testing.assert_allclose(overlap, expected, atol=1e-14)
    np.testing.assert_allclose(overlap, overlap.T, atol=0.0)


def test_large_common_mode_shift_does_not_cause_cancellation():
    lam = np.full((2, 3), 1.0e8)
    lam[:, 1] += np.array([1.0, -1.0])
    lam[:, 2] += np.array([2.0, 1.0])
    expected = np.array(
        [
            [1.0, np.exp(-1.0), np.exp(-2.5)],
            [np.exp(-1.0), 1.0, np.exp(-2.5)],
            [np.exp(-2.5), np.exp(-2.5), 1.0],
        ]
    )

    np.testing.assert_allclose(lf_franck_condon(lam), expected, atol=1e-14)


@pytest.mark.parametrize(
    "lam",
    [np.array(0.0), np.zeros(3), np.zeros((1, 2, 3))],
)
def test_rejects_non_matrix_input(lam):
    with pytest.raises(ValueError, match="2-dimensional"):
        lf_franck_condon(lam)
