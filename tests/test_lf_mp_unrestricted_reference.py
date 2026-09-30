"""Cross-sector checks for unrestricted LF-HF and LF-MP2."""

import numpy as np
import pytest

from src.lf_mp import (
    lf_hf_multi_fixed_energy,
    lf_hf_multi_optimize,
    lf_mp2_reference_point,
    lf_mp2_unrestricted_reference_point,
)


def hopping(nsite):
    if nsite == 2:
        return -np.array([[0.0, 1.0], [1.0, 0.0]])
    eye = np.eye(nsite)
    return -(np.roll(eye, 1, axis=1) + np.roll(eye, -1, axis=1))


def optimize(nsite, nelec, U, g, omega, charge_order=False):
    coeff_a = np.eye(nsite)
    coeff_b = np.eye(nsite)
    if nsite == 4 and nelec == (2, 2):
        coeff_a = np.eye(4)[:, [0, 2, 1, 3]]
        coeff_b = coeff_a.copy() if charge_order else np.eye(4)[:, [1, 3, 0, 2]]
    lam0 = np.eye(nsite) * g / omega
    return lf_hf_multi_optimize(
        hopping(nsite), U, g, omega, nelec, coeff_a, coeff_b, lam0,
    )


def test_general_optimizer_preserves_reference_and_shapes():
    for nsite, nelec, U, g, omega in (
        (4, (2, 2), 4.0, 0.0, 0.5),
        (3, (1, 0), 0.0, 0.4, 0.5),
        (2, (1, 1), 2.0, 0.3, 1.0),
    ):
        result = optimize(nsite, nelec, U, g, omega)
        nrot = sum(n * (nsite - n) for n in nelec)
        assert result.success
        assert result.x.shape == (nrot + nsite**2,)
        assert result.mo_coeff.shape == (2, nsite, nsite)
        assert result.mo_energy.shape == (2, nsite)
        assert result.spin_density.shape == (2, nsite)
        np.testing.assert_allclose(
            result.mo_coeff @ result.mo_coeff.transpose(0, 2, 1),
            np.broadcast_to(np.eye(nsite), (2, nsite, nsite)),
            atol=1e-10,
        )
        energy, density = lf_hf_multi_fixed_energy(
            hopping(nsite), U, g, omega, nelec,
            result.mo_coeff[0, :, :nelec[0]],
            result.mo_coeff[1, :, :nelec[1]],
            result.lam, result.shift,
        )
        assert energy == pytest.approx(result.fun, abs=1e-9)
        np.testing.assert_allclose(density, result.spin_density, atol=1e-10)


def test_zero_coupling_four_electron_ump2_anchor():
    result = optimize(4, (2, 2), 4.0, 0.0, 0.5)
    mp2 = lf_mp2_unrestricted_reference_point(
        hopping(4), 4.0, 0.0, 0.5, (2, 2),
        result.mo_coeff, result.mo_energy, result.lam, result.shift,
        max_total=8,
    )
    assert result.fun == pytest.approx(-1.7632978285545955, abs=1e-8)
    assert mp2["mp2_correction"] == pytest.approx(
        -0.06697357381686747, abs=1e-7,
    )
    assert mp2["zero_double_correction"] == pytest.approx(
        mp2["mp2_correction"], abs=1e-9,
    )
    assert mp2["phonon_pure_correction"] == pytest.approx(0.0, abs=1e-12)


def test_single_electron_agrees_with_existing_reference_at_nonzero_coupling():
    g, omega = 0.3, 0.5
    result = optimize(3, (1, 0), 0.0, g, omega)
    mp2 = lf_mp2_unrestricted_reference_point(
        hopping(3), 0.0, g, omega, (1, 0),
        result.mo_coeff, result.mo_energy, result.lam, result.shift,
        max_total=4, max_excited_modes=3,
    )
    old = lf_mp2_reference_point(
        hopping(3), g, omega, result.shift, result.lam, max_total=4,
    )
    assert mp2["hf_energy"] == pytest.approx(old["hf_energy"], abs=1e-10)
    assert mp2["mp2_correction"] == pytest.approx(
        old["mp2_correction"], abs=1e-12,
    )
    assert mp2["zero_double_count"] == 0


def test_nonuniform_four_electron_reference_and_phonon_cutoff():
    omega, alpha = 0.5, 3.2
    g = np.sqrt(alpha * omega)
    result = optimize(4, (2, 2), 4.0, g, omega, charge_order=True)
    assert result.success
    assert np.ptp(result.spin_density.sum(axis=0)) > 1.0
    corrections = []
    for max_total in (8, 10, 12):
        mp2 = lf_mp2_unrestricted_reference_point(
            hopping(4), 4.0, g, omega, (2, 2),
            result.mo_coeff, result.mo_energy, result.lam, result.shift,
            max_total=max_total,
        )
        assert np.count_nonzero(mp2["occupations"], axis=1).max() <= 2
        corrections.append(mp2["mp2_correction"])
    assert corrections[0] >= corrections[1] >= corrections[2]


def test_nonpositive_denominator_is_rejected():
    zero = np.zeros((2, 2))
    coeff = np.stack((np.eye(2), np.eye(2)))
    energies = np.zeros((2, 2))
    with pytest.raises(ValueError, match="denominator"):
        lf_mp2_unrestricted_reference_point(
            zero, 0.0, 0.0, 1.0, (1, 0),
            coeff, energies, zero, np.zeros(2), max_total=1,
        )
