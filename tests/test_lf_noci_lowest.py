"""Acceptance checks for the lowest NOCI generalized eigenpair."""

import numpy as np
import pytest
from scipy import linalg

from src import lf_mr


@pytest.fixture
def lowest():
    if not hasattr(lf_mr, "lf_noci_lowest"):
        pytest.skip("Student task pending: implement lf_noci_lowest")
    return lf_mr.lf_noci_lowest


def _distinct_kernels():
    tmat = np.array([[0.1, -1.0], [-1.0, 0.2]])
    lam = np.zeros((2, 2, 2))
    lam[1] = np.diag([0.35, 0.45])
    shift = np.zeros((2, 2))
    hmat = lf_mr.lf_frame_hamiltonian(tmat, 0.5, 0.7, lam, shift)
    smat = lf_mr.lf_frame_overlap(lam, shift)
    return hmat, smat


def test_distinct_reference_case_is_positive_definite():
    _, smat = _distinct_kernels()
    assert np.linalg.eigvalsh(smat.reshape(4, 4)).min() > 0.04


def test_single_four_site_frame_has_correct_free_energy(lowest):
    tmat = np.zeros((4, 4))
    for p in range(4):
        tmat[p, (p + 1) % 4] = -1.0
        tmat[(p + 1) % 4, p] = -1.0
    lam = np.zeros((1, 4, 4))
    shift = np.zeros((1, 4))
    hmat = lf_mr.lf_frame_hamiltonian(tmat, 0.0, 0.5, lam, shift)
    smat = lf_mr.lf_frame_overlap(lam, shift)
    energy, coeff, rank = lowest(hmat, smat)
    assert energy == pytest.approx(-2.0, abs=1e-12)
    assert coeff.shape == (1, 4)
    assert rank == 4
    c = coeff.ravel()
    h, s = hmat.reshape(4, 4), smat.reshape(4, 4)
    np.testing.assert_allclose(c @ s @ c, 1.0, atol=1e-12, rtol=0)
    assert np.linalg.norm(h @ c - energy * (s @ c)) < 1e-10


def test_exact_duplicate_frames_keep_rank_and_energy(lowest):
    tmat = np.zeros((4, 4))
    for p in range(4):
        tmat[p, (p + 1) % 4] = -1.0
        tmat[(p + 1) % 4, p] = -1.0
    lam = np.zeros((2, 4, 4))
    shift = np.zeros((2, 4))
    hmat = lf_mr.lf_frame_hamiltonian(tmat, 0.0, 0.5, lam, shift)
    smat = lf_mr.lf_frame_overlap(lam, shift)
    energy, coeff, rank = lowest(hmat, smat)
    assert rank == 4
    assert energy == pytest.approx(-2.0, abs=1e-12)
    c = coeff.ravel()
    h, s = hmat.reshape(8, 8), smat.reshape(8, 8)
    np.testing.assert_allclose(c @ s @ c, 1.0, atol=1e-12, rtol=0)
    assert np.linalg.norm(h @ c - energy * (s @ c)) < 1e-10


def test_distinct_frames_match_independent_generalized_solver(lowest):
    hmat, smat = _distinct_kernels()
    old_h, old_s = hmat.copy(), smat.copy()
    h, s = hmat.reshape(4, 4), smat.reshape(4, 4)
    expected = linalg.eigh(h, s, subset_by_index=[0, 0])[0][0]
    energy, coeff, rank = lowest(hmat, smat)
    assert rank == 4
    assert coeff.shape == (2, 2)
    np.testing.assert_allclose(energy, expected, atol=1e-12, rtol=0)
    c = coeff.ravel()
    np.testing.assert_allclose(c @ s @ c, 1.0, atol=1e-12, rtol=0)
    assert np.linalg.norm(h @ c - energy * (s @ c)) < 1e-10
    for a in range(2):
        assert energy <= np.linalg.eigvalsh(hmat[a, :, a, :])[0] + 1e-12
    np.testing.assert_array_equal(hmat, old_h)
    np.testing.assert_array_equal(smat, old_s)


def test_nearly_duplicate_physical_frames_follow_cutoff(lowest):
    tmat = np.zeros((1, 1))
    lam = np.zeros((2, 1, 1))
    shift = np.array([[0.0], [1e-6]])
    hmat = lf_mr.lf_frame_hamiltonian(tmat, 0.0, 0.5, lam, shift)
    smat = lf_mr.lf_frame_overlap(lam, shift)
    energy, coeff, rank = lowest(hmat, smat)
    assert rank == 1
    assert np.isfinite(energy)
    c = coeff.ravel()
    np.testing.assert_allclose(
        c @ smat.reshape(2, 2) @ c, 1.0, atol=1e-12, rtol=0
    )
    _, _, finer_rank = lowest(hmat, smat, overlap_cut=1e-14)
    assert finer_rank == 2


def test_rejects_substantially_negative_overlap_eigenvalue(lowest):
    hmat = np.zeros((1, 2, 1, 2))
    smat = np.diag([1.0, -0.1]).reshape(1, 2, 1, 2)
    with pytest.raises(ValueError):
        lowest(hmat, smat)
