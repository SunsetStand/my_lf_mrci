"""Stage-one acceptance checks; missing student implementation is explicit."""

import importlib
import importlib.util

import numpy as np
import pytest


@pytest.fixture
def overlap():
    if importlib.util.find_spec("src.lf_mr") is None:
        pytest.skip("Student task pending: implement src/lf_mr.py")
    module = importlib.import_module("src.lf_mr")
    return module.lf_frame_overlap


def coherent_product(eta, nmax):
    """Project an infinite coherent state into a per-mode Fock cutoff."""
    result = np.ones(1)
    for displacement in eta:
        mode = np.empty(nmax + 1)
        mode[0] = np.exp(-0.5 * displacement**2)
        for n in range(1, nmax + 1):
            mode[n] = mode[n - 1] * displacement / np.sqrt(n)
        result = np.kron(result, mode)
    return result


def fock_gram(lam, shift, nmax):
    """Build explicit site-times-Fock vectors independently of the kernel."""
    k, nmode, length = lam.shape
    vectors = np.zeros((k * length, length * (nmax + 1)**nmode))
    for a in range(k):
        for p in range(length):
            electron = np.eye(length)[p]
            phonon = coherent_product(shift[a] - lam[a, :, p], nmax)
            vectors[a * length + p] = np.kron(electron, phonon)
    return vectors @ vectors.T


def test_fock_helper_keeps_projection_norm_and_signed_amplitudes():
    eta = np.array([-0.8, 0.5])
    vacuum = coherent_product(eta, 0)
    np.testing.assert_allclose(vacuum, [np.exp(-0.5 * (eta @ eta))])
    assert vacuum @ vacuum < 1.0
    low = coherent_product(eta, 2).reshape(3, 3)
    assert low[1, 0] < 0.0
    assert low[0, 1] > 0.0
    high = coherent_product(eta, 16)
    np.testing.assert_allclose(high @ high, 1.0, atol=2e-14, rtol=0)


def test_single_frame_identity_shape_dtype_and_no_mutation(overlap):
    lam = np.arange(16, dtype=np.float64).reshape(1, 4, 4) / 10
    shift = np.array([[0.3, -0.2, 0.1, 0.4]])
    originals = lam.copy(), shift.copy()
    s = overlap(lam, shift, nelec=(1, 0))
    assert s.shape == (1, 4, 1, 4)
    assert s.dtype == np.float64
    np.testing.assert_array_equal(s.reshape(4, 4), np.eye(4))
    np.testing.assert_array_equal(lam, originals[0])
    np.testing.assert_array_equal(shift, originals[1])


def test_repeated_frames_have_rank_l(overlap):
    lam = np.tile(np.diag([0.2, -0.3, 0.5, 0.7]), (2, 1, 1))
    shift = np.tile([0.1, -0.1, 0.2, 0.3], (2, 1))
    s = overlap(lam, shift).reshape(8, 8)
    np.testing.assert_array_equal(s, np.tile(np.eye(4), (2, 2)))
    assert np.linalg.matrix_rank(s, tol=1e-12) == 4


def test_known_displacements_and_electron_orthogonality(overlap):
    lam = np.zeros((2, 2, 2))
    lam[1] = [[1.0, -2.0], [-1.0, 0.5]]
    s = overlap(lam, np.zeros((2, 2)))
    np.testing.assert_allclose(
        s[0, :, 1, :], np.diag(np.exp([-1.0, -2.125])), atol=1e-14, rtol=0
    )
    np.testing.assert_array_equal(s[:, 0, :, 1], np.zeros((2, 2)))
    np.testing.assert_array_equal(s[:, 1, :, 0], np.zeros((2, 2)))


def test_gauge_invariance_symmetry_and_positive_semidefiniteness(overlap):
    rng = np.random.default_rng(704)
    lam = rng.normal(size=(3, 4, 4))
    shift = rng.normal(size=(3, 4))
    gauge = rng.normal(size=(3, 4))
    s = overlap(lam, shift)
    changed = overlap(lam + gauge[:, :, None], shift + gauge)
    np.testing.assert_allclose(changed, s, atol=1e-14, rtol=0)
    matrix = s.reshape(12, 12)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-14, rtol=0)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-12


def test_independent_fock_projection_converges(overlap):
    lam = np.array([[[0.2, -0.4], [0.6, 0.1]],
                    [[-0.5, 0.3], [0.2, -0.7]]])
    shift = np.array([[0.1, -0.2], [-0.1, 0.2]])
    exact = overlap(lam, shift).reshape(4, 4)
    errors = [np.max(np.abs(fock_gram(lam, shift, n) - exact))
              for n in (0, 4, 16)]
    assert errors[0] > 0.1
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 2e-13


@pytest.mark.parametrize("lam,shift", [
    (np.zeros((0, 2, 2)), np.zeros((0, 2))),
    (np.zeros((1, 0, 0)), np.zeros((1, 0))),
    (np.zeros((2, 2)), np.zeros((1, 2))),
    (np.zeros((1, 2, 3)), np.zeros((1, 2))),
    (np.zeros((1, 2, 2)), np.zeros((2, 2))),
    (np.full((1, 2, 2), np.nan), np.zeros((1, 2))),
    (np.zeros((1, 2, 2)), np.full((1, 2), np.inf)),
])
def test_rejects_invalid_shapes_and_nonfinite_data(overlap, lam, shift):
    with pytest.raises(ValueError):
        overlap(lam, shift)


@pytest.mark.parametrize("dtype", [np.complex128, np.float32, np.int64])
@pytest.mark.parametrize("target", ["lam", "shift"])
def test_rejects_unsupported_dtypes(overlap, dtype, target):
    lam = np.zeros((1, 2, 2))
    shift = np.zeros((1, 2))
    if target == "lam":
        lam = lam.astype(dtype)
    else:
        shift = shift.astype(dtype)
    with pytest.raises(TypeError):
        overlap(lam, shift)
