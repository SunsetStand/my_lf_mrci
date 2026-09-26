"""Acceptance tests for the physical one-electron multi-frame H kernel."""

import importlib

import numpy as np
import pytest

from src.lf_mp import lf_effective_one_body


@pytest.fixture
def hkernel():
    module = importlib.import_module("src.lf_mr")
    if not hasattr(module, "lf_frame_hamiltonian"):
        pytest.skip("Student task pending: implement lf_frame_hamiltonian")
    return module.lf_frame_hamiltonian


def _coherent_vector(displacements, nmax):
    vector = np.ones(1)
    for amplitude in displacements:
        mode = np.empty(nmax + 1)
        mode[0] = np.exp(-amplitude**2 / 2)
        for n in range(1, nmax + 1):
            mode[n] = mode[n - 1] * amplitude / np.sqrt(n)
        vector = np.kron(vector, mode)
    return vector


def _dense_fock_expectation(tmat, g, omega, lam, shift, nmax):
    """Independent truncated site-times-Fock matrix and state vectors."""
    k, length, _ = lam.shape
    assert length == 2
    d = nmax + 1
    b = np.diag(np.sqrt(np.arange(1, d)), 1)
    eye = np.eye(d)
    bx = [np.kron(b, eye), np.kron(eye, b)]
    phonon_dim = d * d
    number = sum((operator.T @ operator for operator in bx),
                 np.zeros((phonon_dim, phonon_dim)))
    physical_h = np.kron(tmat, np.eye(phonon_dim))
    physical_h += omega * np.kron(np.eye(length), number)
    for p in range(length):
        site = np.zeros((length, length))
        site[p, p] = 1.0
        physical_h += g * np.kron(site, bx[p] + bx[p].T)
    vectors = np.empty((k * length, length * phonon_dim))
    for a in range(k):
        for p in range(length):
            eta = shift[a] - lam[a, :, p]
            vectors[a * length + p] = np.kron(
                np.eye(length)[p], _coherent_vector(eta, nmax)
            )
    return (vectors @ physical_h @ vectors.T).reshape(k, length, k, length)


def test_dense_fock_helper_recovers_vacuum_hopping():
    tmat = np.array([[0.2, -0.8], [-0.8, -0.1]])
    result = _dense_fock_expectation(
        tmat, 0.4, 0.6, np.zeros((1, 2, 2)),
        np.zeros((1, 2)), 3,
    )
    np.testing.assert_array_equal(result[0, :, 0, :], tmat)


def test_hopping_survives_different_electron_sites(hkernel):
    tmat = np.array([[0.15, -0.7], [-0.7, -0.2]])
    lam = np.zeros((2, 2, 2))
    lam[1] = [[-0.4, -1.0], [0.1, 0.5]]
    shift = np.zeros((2, 2))
    g = 0.3
    result = hkernel(tmat, g, 0.6, lam, shift)
    assert result.shape == (2, 2, 2, 2)
    assert result.dtype == np.float64
    np.testing.assert_allclose(
        result[0, 0, 1, 1], -0.7 * np.exp(-0.625), atol=1e-14, rtol=0
    )
    np.testing.assert_allclose(
        result[0, 0, 1, 0], (0.15 + 0.4 * g) * np.exp(-0.085),
        atol=1e-14, rtol=0,
    )


def test_single_frame_matches_existing_lf_effective_one_body(hkernel):
    tmat = np.array([[0.2, -0.8], [-0.8, -0.1]])
    lam = np.array([[[0.2, -0.3], [0.4, 0.1]]])
    shift = np.array([[0.15, -0.25]])
    g = 0.42
    omega = 0.7
    expected = lf_effective_one_body(
        tmat, g, omega, shift[0], lam[0]
    ) + omega * (shift[0] @ shift[0]) * np.eye(2)
    result = hkernel(tmat, g, omega, lam, shift)
    np.testing.assert_allclose(result[0, :, 0, :], expected,
                               atol=1e-14, rtol=0)


def test_zero_displacement_and_duplicate_frames(hkernel):
    tmat = np.array([[0.3, -1.0], [-1.0, 0.4]])
    result = hkernel(tmat, 0.5, 0.7, np.zeros((2, 2, 2)),
                     np.zeros((2, 2)))
    for a in range(2):
        for b in range(2):
            np.testing.assert_array_equal(result[a, :, b, :], tmat)


def test_rejects_nearly_symmetric_hopping(hkernel):
    tmat = np.array([[0.0, -1.0], [-1.0 + 1e-9, 0.0]])
    with pytest.raises(ValueError, match="symmetric"):
        hkernel(tmat, 0.4, 0.6, np.zeros((1, 2, 2)),
                np.zeros((1, 2)))


def test_gauge_invariance_hermiticity_and_input_ownership(hkernel):
    rng = np.random.default_rng(93)
    tmat = np.array([[0.2, -0.9], [-0.9, -0.1]])
    lam = rng.normal(scale=0.3, size=(3, 2, 2))
    shift = rng.normal(scale=0.2, size=(3, 2))
    old_lam, old_shift = lam.copy(), shift.copy()
    gauge = rng.normal(size=(3, 2))
    result = hkernel(tmat, 0.4, 0.6, lam, shift)
    shifted = hkernel(tmat, 0.4, 0.6,
                      lam + gauge[:, :, None], shift + gauge)
    np.testing.assert_allclose(result, shifted, atol=2e-14, rtol=0)
    matrix = result.reshape(6, 6)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(lam, old_lam)
    np.testing.assert_array_equal(shift, old_shift)


def test_independent_dense_fock_projection_converges(hkernel):
    tmat = np.array([[0.1, -0.8], [-0.8, -0.2]])
    lam = np.array([
        [[0.2, -0.3], [0.1, 0.25]],
        [[-0.25, 0.15], [0.2, -0.1]],
    ])
    shift = np.array([[0.1, -0.05], [-0.1, 0.1]])
    g, omega = 0.4, 0.6
    exact = hkernel(tmat, g, omega, lam, shift)
    low = _dense_fock_expectation(tmat, g, omega, lam, shift, 0)
    high = _dense_fock_expectation(tmat, g, omega, lam, shift, 10)
    low_error = np.max(np.abs(low - exact))
    high_error = np.max(np.abs(high - exact))
    assert low_error > 0.01
    assert high_error < low_error
    assert high_error < 2e-13
