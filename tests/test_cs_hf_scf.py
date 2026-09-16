"""Acceptance tests for the one-electron CS-HF SCF driver."""

import numpy as np
import pytest

from src.cs_mp import cs_fock, cs_hf_scf
from src.my_direct_ep import electron_ring_hopping


def _eigen_residual(
    fock: np.ndarray,
    coeff: np.ndarray,
) -> tuple[float, float]:
    """Return the Rayleigh quotient and its eigenvector residual norm."""
    orbital_energy = float(np.vdot(coeff, fock @ coeff).real)
    residual = float(np.linalg.norm(fock @ coeff - orbital_energy * coeff))
    return orbital_energy, residual


def test_zero_coupling_recovers_ring_ground_state():
    hopping = electron_ring_hopping(4, -1.0)

    energy, coeff, shift, converged, niter, density_residual = cs_hf_scf(
        hopping,
        g=0.0,
        omega=0.5,
        coeff0=np.array([1.0, 0.0, 0.0, 0.0]),
    )

    fock = cs_fock(hopping, 0.0, shift)
    orbital_energy, eigen_residual = _eigen_residual(fock, coeff)

    assert converged
    assert niter == 2
    assert density_residual < 1e-10
    np.testing.assert_allclose(np.linalg.norm(coeff), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.abs(coeff) ** 2, 0.25, atol=1e-14)
    np.testing.assert_allclose(shift, 0.0, atol=1e-14)
    np.testing.assert_allclose(energy, -2.0, atol=1e-14)
    np.testing.assert_allclose(orbital_energy, -2.0, atol=1e-14)
    assert eigen_residual < 1e-10


def test_uniform_solution_distinguishes_total_and_orbital_energies():
    omega = 0.5
    alpha = 0.8
    g = np.sqrt(alpha * omega)
    hopping = electron_ring_hopping(4, -1.0)

    energy, coeff, shift, converged, niter, density_residual = cs_hf_scf(
        hopping,
        g,
        omega,
        np.full(4, 0.5),
    )

    fock = cs_fock(hopping, g, shift)
    orbital_energy, eigen_residual = _eigen_residual(fock, coeff)

    assert converged
    assert niter == 1
    assert type(energy) is float
    assert type(density_residual) is float
    assert density_residual < 1e-10
    np.testing.assert_allclose(np.abs(coeff) ** 2, 0.25, atol=1e-14)
    np.testing.assert_allclose(shift, -g / (4.0 * omega), atol=1e-14)
    np.testing.assert_allclose(energy, -2.0 - alpha / 4.0, atol=1e-14)
    np.testing.assert_allclose(orbital_energy, -2.0 - alpha / 2.0, atol=1e-14)
    assert eigen_residual < 1e-10


def test_scaled_uniform_guess_is_normalized_before_first_cycle():
    omega = 0.5
    alpha = 0.8
    g = np.sqrt(alpha * omega)
    hopping = electron_ring_hopping(4, -1.0)

    energy, coeff, shift, converged, niter, density_residual = cs_hf_scf(
        hopping,
        g,
        omega,
        7.0 * np.full(4, 0.5),
        max_cycle=1,
    )

    assert converged
    assert niter == 1
    assert density_residual < 1e-10
    np.testing.assert_allclose(np.linalg.norm(coeff), 1.0, atol=1e-14)
    np.testing.assert_allclose(shift, -g / (4.0 * omega), atol=1e-14)
    np.testing.assert_allclose(energy, -2.0 - alpha / 4.0, atol=1e-14)


def test_zero_initial_guess_is_rejected():
    hopping = electron_ring_hopping(4, -1.0)

    with pytest.raises(ValueError, match="zero vector"):
        cs_hf_scf(hopping, 0.0, 0.5, np.zeros(4))
