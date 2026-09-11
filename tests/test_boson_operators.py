"""Acceptance tests for projected single-site boson operators."""

import numpy as np
import pytest

from src.my_direct_ep import boson_operators


@pytest.mark.parametrize("nmax", [0, 1, 2, 5])
def test_boson_operators(nmax):
    d = nmax + 1
    b, bdag, number = boson_operators(nmax)
    basis = np.eye(d)
    for op in (b, bdag, number):
        assert op.shape == (d, d)
        assert op.dtype == np.float64
    np.testing.assert_allclose(bdag, b.T.conj())
    np.testing.assert_allclose(number, bdag @ b)
    np.testing.assert_allclose(number, np.diag(np.arange(d)), atol=1e-12)
    for n in range(d):
        down, up = np.zeros(d), np.zeros(d)
        if n > 0:
            down[n - 1] = np.sqrt(n)
        if n < nmax:
            up[n + 1] = np.sqrt(n + 1)
        np.testing.assert_allclose(b @ basis[:, n], down, atol=1e-12)
        np.testing.assert_allclose(bdag @ basis[:, n], up, atol=1e-12)
    commutator = np.eye(d)
    commutator[-1, -1] -= d
    np.testing.assert_allclose(b @ bdag - bdag @ b, commutator, atol=1e-12)


@pytest.mark.parametrize("invalid", [-1, 1.5])
def test_invalid_cutoff(invalid):
    with pytest.raises(ValueError):
        boson_operators(invalid)
