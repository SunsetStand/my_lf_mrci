"""Acceptance checks for the student's multi-electron LF Hamiltonian kernel."""

import numpy as np
import pytest

from src import lf_mr
from src import my_direct_ep as ep
from src.lf_mp import lf_hf_multi_fixed_energy


@pytest.fixture
def hkernel():
    function = lf_mr.lf_multi_frame_hamiltonian
    try:
        function(np.zeros((1, 1)), 0.0, 0.0, 1.0,
                 np.zeros((1, 1, 1)), np.zeros((1, 1)), (1, 0))
    except NotImplementedError:
        pytest.skip("Student task pending: implement lf_multi_frame_hamiltonian")
    return function


def _coherent(eta, nmax):
    result = np.ones(1)
    for z in eta:
        mode = np.empty(nmax + 1)
        mode[0] = np.exp(-0.5 * z*z)
        for n in range(1, nmax + 1):
            mode[n] = mode[n-1] * z / np.sqrt(n)
        result = np.kron(result, mode)
    return result


def _two_electron_fock_kernel(tmat, U, g, omega, lam, shift, nmax):
    """Independent L=2, (1,1) Kronecker Hamiltonian and projected states."""
    d = nmax + 1
    b = np.diag(np.sqrt(np.arange(1, d)), 1)
    bx = (np.kron(b, np.eye(d)), np.kron(np.eye(d), b))
    n0 = np.diag([2.0, 1.0, 1.0, 0.0])
    n1 = np.diag([0.0, 1.0, 1.0, 2.0])
    hel = (np.kron(tmat, np.eye(2)) + np.kron(np.eye(2), tmat)
           + np.diag([U, 0.0, 0.0, U]))
    hph = omega * (bx[0].T @ bx[0] + bx[1].T @ bx[1])
    h = np.kron(hel, np.eye(d*d)) + np.kron(np.eye(4), hph)
    h += g * (np.kron(n0, bx[0] + bx[0].T)
              + np.kron(n1, bx[1] + bx[1].T))
    k = len(lam)
    vectors = np.empty((k*4, 4*d*d))
    for A in range(k):
        for ia in range(2):
            for ib in range(2):
                eta = shift[A] - lam[A, :, ia] - lam[A, :, ib]
                vectors[A*4+ia*2+ib] = np.kron(
                    np.eye(4)[ia*2+ib], _coherent(eta, nmax),
                )
    return (vectors @ h @ vectors.T).reshape(k, 2, 2, k, 2, 2)


def test_independent_fock_helper_at_zero_displacement():
    t = np.array([[0.2, -0.7], [-0.7, -0.1]])
    result = _two_electron_fock_kernel(
        t, 1.3, 0.4, 0.6, np.zeros((1, 2, 2)), np.zeros((1, 2)), 2,
    )
    expected = (np.kron(t, np.eye(2)) + np.kron(np.eye(2), t)
                + np.diag([1.3, 0.0, 0.0, 1.3]))
    np.testing.assert_allclose(result.reshape(4, 4), expected, atol=1e-14)


def test_single_electron_reduction_and_hopping_between_orthogonal_kets(hkernel):
    t = np.array([[0.15, -0.7], [-0.7, -0.2]])
    lam = np.array([[[0.2, -0.1], [0.1, 0.3]],
                    [[-0.2, 0.3], [0.2, -0.1]]])
    shift = np.array([[0.1, -0.05], [-0.1, 0.1]])
    old = lf_mr.lf_frame_hamiltonian(t, 0.4, 0.6, lam, shift)
    result = hkernel(t, 9.0, 0.4, 0.6, lam, shift, (1, 0))
    assert result.dtype == np.float64
    np.testing.assert_allclose(result[:, :, 0, :, :, 0], old, atol=2e-14)
    beta = hkernel(t, 9.0, 0.4, 0.6, lam, shift, (0, 1))
    np.testing.assert_allclose(beta[:, 0, :, :, 0, :], old, atol=2e-14)
    s = lf_mr.lf_multi_frame_overlap(lam, shift, (1, 0))
    assert s[0, 0, 0, 1, 1, 0] == 0.0
    assert abs(result[0, 0, 0, 1, 1, 0]) > 0.1


def test_zero_displacement_matches_existing_electronic_contractions(hkernel):
    L, nelec = 3, (2, 1)
    t = np.array([[0.1, -0.7, -0.2], [-0.7, -0.2, -0.4],
                  [-0.2, -0.4, 0.3]])
    shape = ep.make_shape(L, nelec, 0)
    basis = np.eye(9)
    expected = np.empty((9, 9))
    for column in range(9):
        ket = basis[:, column].reshape(shape)
        acted = ep.contract_1e(t, ket, L, nelec, 0)
        acted += ep.contract_2e_hubbard(1.7, ket, L, nelec, 0)
        expected[:, column] = acted.ravel()
    result = hkernel(t, 1.7, 0.5, 0.6, np.zeros((2, L, L)),
                     np.zeros((2, L)), nelec)
    assert result.shape == (2, 3, 3, 2, 3, 3)
    for A in range(2):
        for B in range(2):
            np.testing.assert_allclose(result[A, :, :, B, :, :].reshape(9, 9),
                                       expected, atol=2e-14)
    # alpha |0,1> -> |1,2>: the hop 0 -> 2 has a negative fermion sign.
    assert result[0, 2, 0, 0, 0, 0] == pytest.approx(0.2)


def test_fixed_slater_state_recovers_existing_lf_hf_energy(hkernel):
    rng = np.random.default_rng(815)
    L, nelec = 3, (2, 1)
    raw = rng.normal(size=(L, L))
    t = (raw + raw.T) / 2
    ca = np.linalg.qr(rng.normal(size=(L, L)))[0][:, :2]
    cb = np.linalg.qr(rng.normal(size=(L, L)))[0][:, :1]
    lam = rng.normal(scale=0.2, size=(1, L, L))
    shift = rng.normal(scale=0.1, size=(1, L))
    def amplitudes(occupied, count):
        strings, _ = ep.make_electron_basis(L, count)
        return np.array([np.linalg.det(occupied[
            [p for p in range(L) if int(bits) & (1 << p)], :
        ]) for bits in strings])
    coeff = np.outer(amplitudes(ca, 2), amplitudes(cb, 1)).ravel()
    result = hkernel(t, 1.4, 0.3, 0.7, lam, shift, nelec)
    expected, _ = lf_hf_multi_fixed_energy(
        t, 1.4, 0.3, 0.7, nelec, ca, cb, lam[0], shift[0],
    )
    assert coeff @ result.reshape(9, 9) @ coeff == pytest.approx(expected, abs=2e-12)


def test_independent_fock_projection_and_hermiticity(hkernel):
    t = np.array([[0.1, -0.8], [-0.8, -0.2]])
    lam = np.array([[[0.2, -0.1], [0.1, 0.15]],
                    [[-0.15, 0.1], [0.2, -0.1]]])
    shift = np.array([[0.1, -0.05], [-0.1, 0.05]])
    originals = t.copy(), lam.copy(), shift.copy()
    result = hkernel(t, 1.3, 0.4, 0.6, lam, shift, (1, 1))
    low = _two_electron_fock_kernel(t, 1.3, 0.4, 0.6, lam, shift, 0)
    high = _two_electron_fock_kernel(t, 1.3, 0.4, 0.6, lam, shift, 10)
    assert np.max(np.abs(low-result)) > 1e-3
    np.testing.assert_allclose(result, high, atol=3e-12, rtol=0)
    np.testing.assert_allclose(result.reshape(8, 8), result.reshape(8, 8).T,
                               atol=2e-14, rtol=0)
    for array, original in zip((t, lam, shift), originals):
        np.testing.assert_array_equal(array, original)


def test_empty_electron_sector_retains_phonon_energy(hkernel):
    shift = np.array([[0.2, -0.1], [-0.1, 0.3]])
    lam = np.ones((2, 2, 2))
    result = hkernel(np.zeros((2, 2)), 3.0, 0.7, 0.6, lam, shift, (0, 0))
    expected = 0.6 * np.sum(shift[0]*shift[1]) * np.exp(
        -0.5 * np.sum((shift[0]-shift[1])**2),
    )
    assert result[0, 0, 0, 1, 0, 0] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize('name,value', [
    ('omega', 0.0), ('omega', -1.0), ('g', np.nan), ('U', np.inf),
    ('U', True), ('g', 1+0j),
])
def test_invalid_scalars_rejected_before_student_body(name, value):
    args = dict(tmat=np.zeros((2, 2)), U=1.0, g=0.3, omega=0.5,
                lam=np.zeros((1, 2, 2)), shift=np.zeros((1, 2)), nelec=(1, 1))
    args[name] = value
    with pytest.raises(ValueError):
        lf_mr.lf_multi_frame_hamiltonian(**args)


def test_invalid_hopping_rejected_before_student_body():
    args = (1.0, 0.3, 0.5, np.zeros((1, 2, 2)), np.zeros((1, 2)), (1, 1))
    with pytest.raises(ValueError):
        lf_mr.lf_multi_frame_hamiltonian(np.zeros((3, 3)), *args)
    with pytest.raises(TypeError):
        lf_mr.lf_multi_frame_hamiltonian(np.zeros((2, 2), dtype=np.float32), *args)
    with pytest.raises(ValueError):
        lf_mr.lf_multi_frame_hamiltonian(np.array([[0., 1.], [0., 0.]]), *args)
