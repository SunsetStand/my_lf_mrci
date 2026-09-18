"""Acceptance tests for the one-electron LF stationary displacement."""

import numpy as np
import pytest

from src.cs_mp import cs_stationary_shift
from src.lf_mp import lf_stationary_shift


def test_zero_lambda_recovers_cs_shift_for_complex_orbital():
    omega = 0.5
    g = 0.4
    coeff = np.full(4, 0.5, dtype=np.complex128) * np.exp(0.37j)
    lam = np.zeros((4, 4))
    original_coeff = coeff.copy()
    original_lam = lam.copy()

    shift = lf_stationary_shift(coeff, lam, g, omega)

    np.testing.assert_allclose(
        shift,
        cs_stationary_shift(coeff, g, omega),
        atol=1e-14,
    )
    np.testing.assert_allclose(shift, -g / (4.0 * omega), atol=1e-14)
    np.testing.assert_array_equal(coeff, original_coeff)
    np.testing.assert_array_equal(lam, original_lam)


def test_localized_canonical_lf_displacement_cancels_cs_shift():
    omega = 0.5
    g = 0.4
    coeff = np.array([1.0, 0.0, 0.0, 0.0])
    lam = (g / omega) * np.eye(4)

    shift = lf_stationary_shift(coeff, lam, g, omega)

    np.testing.assert_allclose(shift, np.zeros(4), atol=1e-14)


def test_common_row_shift_produces_gauge_covariant_stationary_shift():
    rng = np.random.default_rng(9)
    coeff = rng.normal(size=4) + 1j * rng.normal(size=4)
    coeff /= np.linalg.norm(coeff)
    lam = rng.normal(size=(4, 4))
    mode_shift = rng.normal(size=4)

    shift = lf_stationary_shift(coeff, lam, g=0.4, omega=0.5)
    translated_shift = lf_stationary_shift(
        coeff,
        lam + mode_shift[:, None],
        g=0.4,
        omega=0.5,
    )

    np.testing.assert_allclose(
        translated_shift,
        shift + mode_shift,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    ("coeff", "lam"),
    [
        (np.zeros((2, 2)), np.zeros((2, 2))),
        (np.ones(3), np.zeros(3)),
        (np.ones(3), np.zeros((2, 3))),
        (np.ones(4), np.zeros((3, 3))),
    ],
)
def test_rejects_incompatible_shapes(coeff, lam):
    with pytest.raises(ValueError):
        lf_stationary_shift(coeff, lam, g=0.4, omega=0.5)


@pytest.mark.parametrize("omega", [0.0, -0.5])
def test_rejects_nonpositive_frequency(omega):
    with pytest.raises(ValueError, match="positive"):
        lf_stationary_shift(
            np.full(4, 0.5),
            np.zeros((4, 4)),
            g=0.4,
            omega=omega,
        )
