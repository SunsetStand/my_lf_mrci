"""Stage-1 acceptance tests for the one-electron CS-HF primitives."""

import numpy as np

from src.cs_mp import cs_energy, cs_fock, cs_site_density, cs_stationary_shift
from src.my_direct_ep import electron_ring_hopping


def test_uniform_four_site_reference_matches_fig2b_convention():
    omega = 0.5
    alpha = 0.8
    g = np.sqrt(alpha * omega)
    coeff = np.full(4, 0.5)
    hopping = electron_ring_hopping(4, -1.0)

    density = cs_site_density(coeff)
    shift = cs_stationary_shift(coeff, g, omega)
    fock = cs_fock(hopping, g, shift)
    energy = cs_energy(hopping, g, omega, coeff, shift)

    np.testing.assert_allclose(density, np.full(4, 0.25), atol=1e-14)
    np.testing.assert_allclose(shift, np.full(4, -g / (4.0 * omega)))
    np.testing.assert_allclose(fock, hopping - 0.5 * alpha * np.eye(4))
    np.testing.assert_allclose(energy, -2.0 - alpha / 4.0, atol=1e-14)


def test_localized_reference_has_full_polaron_shift():
    omega = 0.5
    alpha = 1.2
    g = np.sqrt(alpha * omega)
    coeff = np.array([1.0, 0.0, 0.0, 0.0])
    hopping = electron_ring_hopping(4, -1.0)

    shift = cs_stationary_shift(coeff, g, omega)
    energy = cs_energy(hopping, g, omega, coeff, shift)

    np.testing.assert_allclose(shift, [-g / omega, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(energy, -alpha, atol=1e-14)


def test_complex_phase_does_not_change_density_shift_or_energy():
    omega = 0.5
    g = 0.3
    hopping = electron_ring_hopping(4, -1.0)
    coeff = np.full(4, 0.5, dtype=np.complex128)
    phased_coeff = np.exp(0.37j) * coeff

    density = cs_site_density(coeff)
    phased_density = cs_site_density(phased_coeff)
    shift = cs_stationary_shift(coeff, g, omega)
    phased_shift = cs_stationary_shift(phased_coeff, g, omega)

    np.testing.assert_allclose(phased_density, density, atol=1e-14)
    np.testing.assert_allclose(phased_shift, shift, atol=1e-14)
    assert isinstance(cs_energy(hopping, g, omega, phased_coeff, phased_shift), float)
    np.testing.assert_allclose(
        cs_energy(hopping, g, omega, phased_coeff, phased_shift),
        cs_energy(hopping, g, omega, coeff, shift),
        atol=1e-14,
    )
