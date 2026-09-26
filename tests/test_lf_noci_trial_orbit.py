"""Acceptance tests for the energy gained by adding one translated LF orbit."""

import numpy as np
import pytest

from src import lf_mr


@pytest.fixture
def trial_orbit():
    if not hasattr(lf_mr, "lf_noci_trial_orbit"):
        pytest.skip("Student task pending: implement lf_noci_trial_orbit")
    return lf_mr.lf_noci_trial_orbit


def _two_site_model():
    tmat = np.array([[0.0, -1.0], [-1.0, 0.0]])
    return tmat, 0.8, 1.0


def test_candidate_orbit_lowers_energy_through_combined_space(trial_orbit):
    tmat, g, omega = _two_site_model()
    base_lam = np.zeros((1, 2, 2))
    base_shift = np.zeros((1, 2))
    seed_lam = np.diag([0.8, 0.8])
    seed_shift = np.zeros(2)
    inputs = [tmat, base_lam, base_shift, seed_lam, seed_shift]
    copies = [array.copy() for array in inputs]

    e_base, e_aug, rank_base, rank_aug = trial_orbit(
        tmat, g, omega, base_lam, base_shift, seed_lam, seed_shift,
    )

    assert e_base == pytest.approx(-1.0, abs=1e-12)
    assert e_aug == pytest.approx(-1.3263066868550983, abs=1e-12)
    assert rank_base == 2
    assert rank_aug == 4
    assert e_aug < e_base - 0.3
    for array, original in zip(inputs, copies):
        np.testing.assert_array_equal(array, original)


def test_duplicate_candidate_does_not_enlarge_span(trial_orbit):
    tmat, g, omega = _two_site_model()
    seed_lam = np.array([[0.7, 0.15], [-0.2, 0.3]])
    seed_shift = np.array([0.1, -0.05])
    base_lam, base_shift = lf_mr.lf_translation_orbit(
        seed_lam, seed_shift
    )

    e_base, e_aug, rank_base, rank_aug = trial_orbit(
        tmat, g, omega, base_lam, base_shift, seed_lam, seed_shift,
    )

    assert rank_base == 4
    assert rank_aug == rank_base
    assert e_aug == pytest.approx(e_base, abs=1e-12)


def test_full_orbit_is_invariant_to_seed_translation_and_gauge(trial_orbit):
    tmat, g, omega = _two_site_model()
    base_lam = np.zeros((1, 2, 2))
    base_shift = np.zeros((1, 2))
    seed_lam = np.array([[0.6, -0.1], [0.2, 0.4]])
    seed_shift = np.array([0.1, -0.2])
    gauge = np.array([0.25, -0.4])

    baseline = trial_orbit(
        tmat, g, omega, base_lam, base_shift, seed_lam, seed_shift,
    )
    translated = trial_orbit(
        tmat, g, omega, base_lam, base_shift,
        np.roll(seed_lam, 1, axis=(0, 1)),
        np.roll(seed_shift, 1),
    )
    regauged = trial_orbit(
        tmat, g, omega, base_lam, base_shift,
        seed_lam + gauge[:, None],
        seed_shift + gauge,
    )

    assert baseline[2:] == translated[2:] == regauged[2:]
    np.testing.assert_allclose(baseline[:2], translated[:2], atol=1e-12, rtol=0)
    np.testing.assert_allclose(baseline[:2], regauged[:2], atol=1e-12, rtol=0)
