"""Acceptance tests for fixed-parameter local-diagonal LF-HF states."""

import numpy as np
import pytest

from src.lf_mp import (
    lf_effective_one_body,
    lf_energy,
    lf_hf_local_state,
)
from src.my_direct_ep import electron_ring_hopping


def test_zero_coupling_recovers_ring_ground_state():
    hopping = electron_ring_hopping(4, -1.0)

    total_energy, coeff, orbital_energy = lf_hf_local_state(
        np.zeros(8),
        hopping,
        g=0.0,
        omega=0.5,
    )

    np.testing.assert_allclose(total_energy, -2.0, atol=1e-14)
    np.testing.assert_allclose(orbital_energy, -2.0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(coeff), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.abs(coeff) ** 2, 0.25, atol=1e-14)


def test_cs_point_distinguishes_total_and_orbital_energies():
    hopping = electron_ring_hopping(4, -1.0)
    omega = 0.5
    alpha = 0.8
    g = np.sqrt(alpha * omega)
    ell = np.zeros(4)
    shift = np.full(4, -g / (4.0 * omega))

    total_energy, coeff, orbital_energy = lf_hf_local_state(
        np.concatenate((ell, shift)),
        hopping,
        g,
        omega,
    )

    assert type(total_energy) is float
    assert type(orbital_energy) is float
    np.testing.assert_allclose(total_energy, -2.0 - alpha / 4.0, atol=1e-14)
    np.testing.assert_allclose(orbital_energy, -2.0 - alpha / 2.0, atol=1e-14)
    np.testing.assert_allclose(np.abs(coeff) ** 2, 0.25, atol=1e-14)


def test_uniform_canonical_lf_point_has_dressed_ring_energy():
    hopping = electron_ring_hopping(4, -1.0)
    omega = 0.5
    alpha = 0.8
    g = np.sqrt(alpha * omega)
    displacement = g / omega
    params = np.concatenate((np.full(4, displacement), np.zeros(4)))

    total_energy, coeff, orbital_energy = lf_hf_local_state(
        params,
        hopping,
        g,
        omega,
    )
    expected = -alpha - 2.0 * np.exp(-(displacement**2))

    np.testing.assert_allclose(total_energy, expected, atol=1e-14)
    np.testing.assert_allclose(orbital_energy, expected, atol=1e-14)
    np.testing.assert_allclose(np.abs(coeff) ** 2, 0.25, atol=1e-14)


def test_arbitrary_parameters_match_energy_primitive_and_eigen_equation():
    rng = np.random.default_rng(23)
    hopping = electron_ring_hopping(4, -1.0)
    params = rng.normal(size=8)
    original_params = params.copy()
    ell = params[:4]
    shift = params[4:]
    lam = np.diag(ell)

    total_energy, coeff, orbital_energy = lf_hf_local_state(
        params,
        hopping,
        g=0.4,
        omega=0.5,
    )
    reference = lf_energy(hopping, 0.4, 0.5, coeff, shift, lam)
    effective = lf_effective_one_body(hopping, 0.4, 0.5, shift, lam)
    residual = np.linalg.norm(effective @ coeff - orbital_energy * coeff)

    np.testing.assert_allclose(total_energy, reference, atol=1e-14)
    assert residual < 1e-10
    np.testing.assert_array_equal(params, original_params)


@pytest.mark.parametrize(
    ("params", "tmat"),
    [
        (np.zeros(8), np.zeros(4)),
        (np.zeros(8), np.zeros((3, 4))),
        (np.zeros((2, 4)), np.zeros((4, 4))),
        (np.zeros(7), np.zeros((4, 4))),
    ],
)
def test_rejects_invalid_parameter_or_hopping_shapes(params, tmat):
    with pytest.raises(ValueError):
        lf_hf_local_state(params, tmat, g=0.4, omega=0.5)


@pytest.mark.parametrize("omega", [0.0, -0.5])
def test_rejects_nonpositive_frequency(omega):
    with pytest.raises(ValueError, match="positive"):
        lf_hf_local_state(
            np.zeros(8),
            np.zeros((4, 4)),
            g=0.4,
            omega=omega,
        )
