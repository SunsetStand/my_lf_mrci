"""Acceptance tests for single-start local-diagonal LF-HF optimization."""

import numpy as np
import pytest

from src.lf_mp import lf_hf_local_optimize, lf_hf_local_state
from src.my_direct_ep import electron_ring_hopping


def test_zero_coupling_converges_without_modifying_initial_parameters():
    hopping = electron_ring_hopping(4, -1.0)
    params0 = np.random.default_rng(23).normal(scale=0.05, size=8)
    original_params = params0.copy()

    result = lf_hf_local_optimize(params0, hopping, g=0.0, omega=0.5)

    assert result.success
    np.testing.assert_allclose(result.fun, -2.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(result.coeff) ** 2, 0.25, atol=1e-10)
    assert result.shift_residual < 1e-8
    assert type(result.orbital_energy) is float
    assert type(result.shift_residual) is float
    np.testing.assert_array_equal(params0, original_params)


def test_weak_coupling_cs_start_reaches_stationary_lf_solution():
    hopping = electron_ring_hopping(4, -1.0)
    omega = 0.5
    alpha = 0.8
    g = np.sqrt(alpha * omega)
    params0 = np.concatenate((np.zeros(4), np.full(4, -g / (4.0 * omega))))

    result = lf_hf_local_optimize(params0, hopping, g, omega)
    state_energy, coeff, orbital_energy = lf_hf_local_state(result.x, hopping, g, omega)

    assert result.success
    assert result.fun < -2.0 - alpha / 4.0
    np.testing.assert_allclose(result.fun, -2.2964211620316366, atol=1e-11)
    np.testing.assert_allclose(result.fun, state_energy, atol=1e-14)
    np.testing.assert_allclose(result.coeff, coeff, atol=1e-14)
    np.testing.assert_allclose(result.orbital_energy, orbital_energy, atol=1e-14)
    np.testing.assert_allclose(np.abs(result.coeff) ** 2, 0.25, atol=1e-10)
    assert result.shift_residual < 1e-8


@pytest.mark.parametrize(
    "params0",
    [
        np.zeros((2, 4)),
        np.array([0.0] * 7 + [np.nan]),
        np.array([0.0] * 7 + [np.inf]),
        np.array([0.0] * 7 + [1.0j]),
    ],
)
def test_rejects_invalid_initial_parameters(params0):
    with pytest.raises(ValueError):
        lf_hf_local_optimize(params0, np.zeros((4, 4)), g=0.4, omega=0.5)


@pytest.mark.parametrize(("gtol", "max_cycle"), [(0.0, 500), (1e-8, 0)])
def test_rejects_nonpositive_optimizer_controls(gtol, max_cycle):
    with pytest.raises(ValueError, match="positive"):
        lf_hf_local_optimize(
            np.zeros(8),
            np.zeros((4, 4)),
            g=0.4,
            omega=0.5,
            gtol=gtol,
            max_cycle=max_cycle,
        )


def test_delegates_model_shape_and_frequency_validation():
    with pytest.raises(ValueError):
        lf_hf_local_optimize(
            np.zeros(7),
            np.zeros((4, 4)),
            g=0.4,
            omega=0.5,
        )
    with pytest.raises(ValueError, match="positive"):
        lf_hf_local_optimize(
            np.zeros(8),
            np.zeros((4, 4)),
            g=0.4,
            omega=0.0,
        )
