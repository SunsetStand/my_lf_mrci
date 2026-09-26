"""Acceptance tests for fixed-base scoring of candidate LF translation orbits."""

import numpy as np
import pytest

from src import lf_mr


@pytest.fixture
def score_orbits():
    if not hasattr(lf_mr, "lf_noci_score_orbits"):
        pytest.skip("Student task pending: implement lf_noci_score_orbits")
    return lf_mr.lf_noci_score_orbits


def _two_site_case():
    tmat = np.array([[0.0, -1.0], [-1.0, 0.0]])
    base_lam = np.zeros((1, 2, 2))
    base_shift = np.zeros((1, 2))
    return tmat, base_lam, base_shift


def test_scores_independent_candidates_without_updating_base(score_orbits):
    tmat, base_lam, base_shift = _two_site_case()
    candidate_lam = np.stack((
        np.zeros((2, 2)), np.diag([0.4, 0.4]), np.diag([0.8, 0.8]),
    ))
    candidate_shift = np.zeros((3, 2))
    arrays = [tmat, base_lam, base_shift, candidate_lam, candidate_shift]
    originals = [array.copy() for array in arrays]

    e_base, rank_base, energies, ranks = score_orbits(
        tmat, 0.8, 1.0, base_lam, base_shift,
        candidate_lam, candidate_shift,
    )

    assert e_base == pytest.approx(-1.0, abs=1e-12)
    assert rank_base == 2
    assert energies.shape == (3,)
    assert energies.dtype == np.float64
    assert ranks.shape == (3,)
    assert ranks.dtype == np.int64
    np.testing.assert_allclose(
        energies,
        [-1.0, -1.332547205409442, -1.3263066868550983],
        atol=1e-12, rtol=0,
    )
    np.testing.assert_array_equal(ranks, [2, 4, 4])
    for array, original in zip(arrays, originals):
        np.testing.assert_array_equal(array, original)


def test_candidate_order_only_permutes_candidate_scores(score_orbits):
    tmat, base_lam, base_shift = _two_site_case()
    candidate_lam = np.stack((
        np.diag([0.4, 0.4]), np.diag([1.2, 1.2]),
        np.array([[0.6, -0.1], [0.2, 0.4]]),
    ))
    candidate_shift = np.array([[0.0, 0.0], [0.0, 0.0], [0.1, -0.2]])
    order = np.array([2, 0, 1])

    baseline = score_orbits(
        tmat, 0.8, 1.0, base_lam, base_shift,
        candidate_lam, candidate_shift,
    )
    permuted = score_orbits(
        tmat, 0.8, 1.0, base_lam, base_shift,
        candidate_lam[order], candidate_shift[order],
    )

    assert permuted[0] == pytest.approx(baseline[0], abs=1e-12)
    assert permuted[1] == baseline[1]
    np.testing.assert_allclose(permuted[2], baseline[2][order], atol=1e-12, rtol=0)
    np.testing.assert_array_equal(permuted[3], baseline[3][order])


def test_gauge_and_translation_equivalent_seeds_have_equal_scores(score_orbits):
    tmat, base_lam, base_shift = _two_site_case()
    seed_lam = np.array([[0.6, -0.1], [0.2, 0.4]])
    seed_shift = np.array([0.1, -0.2])
    gauge = np.array([0.25, -0.4])
    candidate_lam = np.stack((
        seed_lam,
        seed_lam + gauge[:, None],
        np.roll(seed_lam, 1, axis=(0, 1)),
    ))
    candidate_shift = np.stack((
        seed_shift,
        seed_shift + gauge,
        np.roll(seed_shift, 1),
    ))

    _, _, energies, ranks = score_orbits(
        tmat, 0.8, 1.0, base_lam, base_shift,
        candidate_lam, candidate_shift,
    )

    np.testing.assert_allclose(energies, np.full(3, energies[0]), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(ranks, np.full(3, ranks[0]))
