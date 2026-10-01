"""Multi-electron NOCI solver checks against independent eigensolvers."""

import numpy as np
import pytest
from pyscf import fci
from scipy import linalg

from src import lf_mr
from src.my_direct_ep import electron_ring_hopping


def _kernels(lam, shift, nelec=(2, 1)):
    t = np.array([[0.1, -0.7, -0.2], [-0.7, -0.2, -0.4],
                  [-0.2, -0.4, 0.3]])
    return (
        lf_mr.lf_multi_frame_hamiltonian(t, 1.4, 0.3, 0.7, lam, shift, nelec),
        lf_mr.lf_multi_frame_overlap(lam, shift, nelec),
    )


def _check_pair(h, s, energy, coeff):
    c = coeff.ravel()
    h, s = h.reshape(c.size, c.size), s.reshape(c.size, c.size)
    assert c @ s @ c == pytest.approx(1.0, abs=2e-12)
    assert np.linalg.norm(h @ c - energy * (s @ c)) < 1e-10


def test_nonzero_coupling_matches_scipy_and_each_single_frame_bound():
    rng = np.random.default_rng(704)
    lam = rng.normal(scale=0.2, size=(2, 3, 3))
    shift = rng.normal(scale=0.1, size=(2, 3))
    h, s = _kernels(lam, shift)
    old_h, old_s = h.copy(), s.copy()
    assert np.linalg.eigvalsh(s.reshape(18, 18)).min() > 1e-3
    expected = linalg.eigh(h.reshape(18, 18), s.reshape(18, 18),
                          subset_by_index=[0, 0])[0][0]
    energy, coeff, rank = lf_mr.lf_noci_lowest(h, s)
    assert isinstance(energy, float)
    assert isinstance(rank, int)
    assert coeff.dtype == np.float64
    assert coeff.shape == (2, 3, 3)
    assert rank == 18
    assert energy == pytest.approx(expected, abs=2e-12)
    _check_pair(h, s, energy, coeff)
    for A in range(2):
        hf, sf = _kernels(lam[A:A+1], shift[A:A+1])
        single_energy = lf_mr.lf_noci_lowest(hf, sf)[0]
        assert energy <= single_energy + 1e-12
    np.testing.assert_array_equal(h, old_h)
    np.testing.assert_array_equal(s, old_s)


def test_duplicate_multi_electron_frames_remove_only_redundancy():
    rng = np.random.default_rng(204)
    lam = rng.normal(scale=0.2, size=(1, 3, 3))
    shift = rng.normal(scale=0.1, size=(1, 3))
    h0, s0 = _kernels(lam, shift)
    expected = lf_mr.lf_noci_lowest(h0, s0)[0]
    h, s = _kernels(np.repeat(lam, 3, axis=0), np.repeat(shift, 3, axis=0))
    energy, coeff, rank = lf_mr.lf_noci_lowest(h, s)
    assert coeff.shape == (3, 3, 3)
    assert rank == 9
    assert energy == pytest.approx(expected, abs=2e-12)
    _check_pair(h, s, energy, coeff)


def test_single_electron_four_and_six_index_interfaces_agree():
    t = np.array([[0.1, -0.7], [-0.7, -0.2]])
    lam = np.zeros((2, 2, 2))
    lam[1] = np.diag([0.35, 0.45])
    shift = np.zeros((2, 2))
    h4 = lf_mr.lf_frame_hamiltonian(t, 0.4, 0.7, lam, shift)
    s4 = lf_mr.lf_frame_overlap(lam, shift)
    e4, c4, rank4 = lf_mr.lf_noci_lowest(h4, s4)
    for nelec in ((1, 0), (0, 1)):
        h6 = lf_mr.lf_multi_frame_hamiltonian(t, 3.0, 0.4, 0.7, lam, shift, nelec)
        s6 = lf_mr.lf_multi_frame_overlap(lam, shift, nelec)
        e6, c6, rank6 = lf_mr.lf_noci_lowest(h6, s6)
        assert e6 == pytest.approx(e4, abs=2e-12)
        assert rank6 == rank4
        assert c6.shape == ((2, 2, 1) if nelec == (1, 0) else (2, 1, 2))
        assert abs(c4.ravel() @ s4.reshape(4, 4) @ c6.ravel()) == pytest.approx(1.0)
        _check_pair(h6, s6, e6, c6)


def test_four_site_four_electron_zero_coupling_matches_electronic_fci():
    L, nelec, U = 4, (2, 2), 4.0
    t = electron_ring_hopping(L)
    eri = np.zeros((L, L, L, L))
    sites = np.arange(L)
    eri[sites, sites, sites, sites] = U
    expected, _ = fci.direct_spin1.kernel(t, eri, L, nelec, tol=1e-13)
    lam, shift = np.zeros((1, L, L)), np.zeros((1, L))
    h = lf_mr.lf_multi_frame_hamiltonian(t, U, 0.0, 0.5, lam, shift, nelec)
    s = lf_mr.lf_multi_frame_overlap(lam, shift, nelec)
    energy, coeff, rank = lf_mr.lf_noci_lowest(h, s)
    assert coeff.shape == (1, 6, 6)
    assert rank == 36
    assert energy == pytest.approx(expected, abs=2e-12)
    assert energy == pytest.approx(-2.102748483462, abs=2e-12)
    _check_pair(h, s, energy, coeff)


def test_truncated_six_index_problem_uses_retained_subspace_residual():
    shape = (2, 1, 1, 2, 1, 1)
    h = np.array([[-1.0, 0.2], [0.2, 0.4]])
    s = np.diag([1.0, 1e-14])
    energy, coeff, rank = lf_mr.lf_noci_lowest(h.reshape(shape), s.reshape(shape))
    residual = h @ coeff.ravel() - energy * (s @ coeff.ravel())
    assert rank == 1
    assert energy == pytest.approx(-1.0)
    assert abs(residual[0]) < 1e-12
    assert np.linalg.norm(residual) > 0.1
    assert coeff.ravel() @ s @ coeff.ravel() == pytest.approx(1.0)


@pytest.mark.parametrize('h,s', [
    (np.zeros((1, 2, 1, 1, 3, 1)), np.zeros((1, 2, 1, 1, 3, 1))),
    (np.zeros((1, 0, 1, 1, 0, 1)), np.zeros((1, 0, 1, 1, 0, 1))),
    (np.zeros((1, 2, 1, 1, 2, 1)), np.zeros((1, 2, 1, 2))),
    (np.zeros((1, 1, 1)), np.zeros((1, 1, 1))),
])
def test_rejects_incompatible_bra_ket_shapes(h, s):
    with pytest.raises(ValueError):
        lf_mr.lf_noci_lowest(h, s)


@pytest.mark.parametrize('cut', [True, 0.0, -1.0, np.nan, np.inf, 1e-10+0j])
def test_rejects_invalid_cutoffs(cut):
    s = np.eye(2).reshape(2, 1, 1, 2, 1, 1)
    with pytest.raises(ValueError):
        lf_mr.lf_noci_lowest(s, s, overlap_cut=cut)


@pytest.mark.parametrize('eigenvalues', [[1.0, -0.1], [0.0, 0.0]])
def test_rejects_indefinite_or_empty_retained_space(eigenvalues):
    shape = (2, 1, 1, 2, 1, 1)
    h = np.eye(2).reshape(shape)
    s = np.diag(eigenvalues).reshape(shape)
    with pytest.raises(ValueError):
        lf_mr.lf_noci_lowest(h, s)
