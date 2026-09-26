"""Acceptance checks for a joint electron-site and phonon-displacement moment."""

import numpy as np
import pytest

from src import lf_mr
from src.my_direct_ep import electron_ring_hopping


@pytest.fixture
def moment():
    if not hasattr(lf_mr, "lf_noci_site_phonon_moment"):
        pytest.skip("Student task pending: implement lf_noci_site_phonon_moment")
    return lf_mr.lf_noci_site_phonon_moment


def _coherent_vector(eta, nmax):
    vector = np.ones(1)
    for amplitude in eta:
        mode = np.empty(nmax + 1)
        mode[0] = np.exp(-amplitude**2 / 2)
        for n in range(1, nmax + 1):
            mode[n] = mode[n - 1] * amplitude / np.sqrt(n)
        vector = np.kron(vector, mode)
    return vector


def _fock_moment(coeff, lam, shift, nmax):
    """Independent projected Fock expectation for a two-site model."""
    k, length = coeff.shape
    assert length == 2
    d = nmax + 1
    b = np.diag(np.sqrt(np.arange(1, d)), 1)
    eye = np.eye(d)
    bx = [np.kron(b, eye), np.kron(eye, b)]
    phonon_dim = d * d
    psi = np.zeros(length * phonon_dim)
    for a in range(k):
        for p in range(length):
            eta = shift[a] - lam[a, :, p]
            psi += coeff[a, p] * np.kron(
                np.eye(length)[p], _coherent_vector(eta, nmax)
            )
    result = np.empty((length, length))
    for x in range(length):
        for p in range(length):
            site = np.zeros((length, length))
            site[p, p] = 1.0
            operator = np.kron(site, bx[x] + bx[x].T)
            result[x, p] = psi @ operator @ psi
    return result


def test_single_frame_has_factor_two_and_correct_mode_site_order(moment):
    lam = np.array([[[0.3, -0.2], [0.4, 0.1]]])
    shift = np.array([[0.1, -0.1]])
    coeff = np.array([[np.sqrt(0.7), np.sqrt(0.3)]])
    smat = lf_mr.lf_frame_overlap(lam, shift)
    old_coeff, old_smat = coeff.copy(), smat.copy()
    result = moment(coeff, smat, lam, shift)
    eta = shift[0, :, None] - lam[0]
    expected = 2 * eta * coeff[0][None, :]**2
    assert result.shape == (2, 2)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, expected, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(coeff, old_coeff)
    np.testing.assert_array_equal(smat, old_smat)


def test_duplicate_frames_require_cross_terms(moment):
    lam = np.full((2, 1, 1), 0.4)
    shift = np.zeros((2, 1))
    coeff = np.full((2, 1), 0.5)
    smat = lf_mr.lf_frame_overlap(lam, shift)
    result = moment(coeff, smat, lam, shift)
    np.testing.assert_allclose(result, [[-0.8]], atol=1e-14, rtol=0)


def test_distinct_frames_agree_with_projected_fock_operator(moment):
    lam = np.array([
        [[0.2, -0.3], [0.1, 0.25]],
        [[-0.25, 0.15], [0.2, -0.1]],
    ])
    shift = np.array([[0.1, -0.05], [-0.1, 0.1]])
    smat = lf_mr.lf_frame_overlap(lam, shift)
    raw = np.array([[0.6, 0.1], [-0.2, 0.45]])
    norm = raw.ravel() @ smat.reshape(4, 4) @ raw.ravel()
    coeff = raw / np.sqrt(norm)
    exact = moment(coeff, smat, lam, shift)
    low = _fock_moment(coeff, lam, shift, 0)
    high = _fock_moment(coeff, lam, shift, 10)
    assert np.max(np.abs(low - exact)) > 0.05
    np.testing.assert_allclose(high, exact, atol=2e-13, rtol=0)


def test_joint_moment_is_invariant_under_frame_gauge(moment):
    rng = np.random.default_rng(907)
    lam = rng.normal(scale=0.3, size=(2, 2, 2))
    shift = rng.normal(scale=0.2, size=(2, 2))
    gauge = rng.normal(scale=0.3, size=(2, 2))
    smat = lf_mr.lf_frame_overlap(lam, shift)
    raw = np.array([[0.6, 0.2], [-0.15, 0.4]])
    coeff = raw / np.sqrt(raw.ravel() @ smat.reshape(4, 4) @ raw.ravel())
    changed_lam = lam + gauge[:, :, None]
    changed_shift = shift + gauge
    changed_smat = lf_mr.lf_frame_overlap(changed_lam, changed_shift)
    baseline = moment(coeff, smat, lam, shift)
    changed = moment(coeff, changed_smat, changed_lam, changed_shift)
    np.testing.assert_allclose(changed, baseline, atol=1e-14, rtol=0)


def test_translation_orbit_has_nontrivial_covariant_cloud(moment):
    rng = np.random.default_rng(906)
    lam = rng.normal(scale=0.4, size=(4, 4))
    shift = rng.normal(scale=0.2, size=4)
    lam_orbit, shift_orbit = lf_mr.lf_translation_orbit(lam, shift)
    smat = lf_mr.lf_frame_overlap(lam_orbit, shift_orbit)
    hmat = lf_mr.lf_frame_hamiltonian(
        electron_ring_hopping(4, -1.0), 0.65, 0.5,
        lam_orbit, shift_orbit,
    )
    _, coeff, rank = lf_mr.lf_noci_lowest(hmat, smat)
    assert rank == 16
    rho = lf_mr.lf_noci_site_density(coeff, smat)
    np.testing.assert_allclose(rho, np.full(4, 0.25),
                               atol=1e-10, rtol=0)
    joint = moment(coeff, smat, lam_orbit, shift_orbit)
    for r in range(1, 4):
        np.testing.assert_allclose(
            joint, np.roll(joint, r, axis=(0, 1)),
            atol=5e-14, rtol=0,
        )
    assert np.ptp(joint) > 0.1
