"""Acceptance tests for the one-electron zero-phonon LF energy."""

import numpy as np
import pytest

from src.cs_mp import cs_energy, cs_stationary_shift
from src.lf_mp import lf_effective_one_body, lf_energy
from src.my_direct_ep import electron_ring_hopping


def test_zero_lambda_recovers_cs_total_not_orbital_energy():
    hopping = electron_ring_hopping(4, -1.0)
    omega = 0.5
    alpha = 0.8
    g = np.sqrt(alpha * omega)
    coeff = np.full(4, 0.5, dtype=np.complex128) * np.exp(0.37j)
    shift = cs_stationary_shift(coeff, g, omega)
    lam = np.zeros((4, 4))

    energy = lf_energy(hopping, g, omega, coeff, shift, lam)
    effective = lf_effective_one_body(hopping, g, omega, shift, lam)
    orbital_energy = float(np.vdot(coeff, effective @ coeff).real)

    assert type(energy) is float
    np.testing.assert_allclose(
        energy,
        cs_energy(hopping, g, omega, coeff, shift),
        atol=1e-14,
    )
    np.testing.assert_allclose(energy, -2.0 - alpha / 4.0, atol=1e-14)
    np.testing.assert_allclose(orbital_energy, -2.0 - alpha / 2.0, atol=1e-14)


def test_localized_canonical_lf_state_has_full_polaron_energy():
    hopping = electron_ring_hopping(4, -1.0)
    omega = 0.5
    alpha = 0.8
    g = np.sqrt(alpha * omega)
    coeff = np.array([1.0, 0.0, 0.0, 0.0])
    shift = np.zeros(4)
    lam = (g / omega) * np.eye(4)

    energy = lf_energy(hopping, g, omega, coeff, shift, lam)

    np.testing.assert_allclose(energy, -alpha, atol=1e-14)


def test_total_energy_is_gauge_invariant_for_normalized_complex_orbital():
    rng = np.random.default_rng(18)
    hopping = electron_ring_hopping(4, -1.0)
    coeff = rng.normal(size=4) + 1j * rng.normal(size=4)
    coeff /= np.linalg.norm(coeff)
    shift = rng.normal(size=4)
    lam = rng.normal(size=(4, 4))
    translation = rng.normal(size=4)
    original_coeff = coeff.copy()
    original_shift = shift.copy()
    original_lam = lam.copy()

    energy = lf_energy(hopping, 0.4, 0.5, coeff, shift, lam)
    translated_energy = lf_energy(
        hopping,
        0.4,
        0.5,
        coeff,
        shift + translation,
        lam + translation[:, None],
    )

    np.testing.assert_allclose(translated_energy, energy, atol=1e-13)
    np.testing.assert_array_equal(coeff, original_coeff)
    np.testing.assert_array_equal(shift, original_shift)
    np.testing.assert_array_equal(lam, original_lam)


@pytest.mark.parametrize(
    "coeff",
    [np.zeros((2, 2)), np.zeros(3)],
)
def test_rejects_incompatible_coefficient_shape(coeff):
    with pytest.raises(ValueError):
        lf_energy(
            np.zeros((4, 4)),
            0.4,
            0.5,
            coeff,
            np.zeros(4),
            np.zeros((4, 4)),
        )
