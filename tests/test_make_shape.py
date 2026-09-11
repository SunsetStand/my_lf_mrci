"""Acceptance tests for the structured electron-phonon CI shape."""

import numpy as np
import pytest

from src.my_direct_ep import make_shape


@pytest.mark.parametrize(
    "nsite, nelec, nmax, expected, dimension",
    [(4, (1, 0), 2, (4, 1, 3, 3, 3, 3), 324),
     (4, (2, 1), 1, (6, 4, 2, 2, 2, 2), 384),
     (4, (0, 0), 0, (1, 1, 1, 1, 1, 1), 1),
     (4, (4, 4), 1, (1, 1, 2, 2, 2, 2), 16),
     (1, (1, 0), 2, (1, 1, 3), 3)],
)
def test_ci_shape(nsite, nelec, nmax, expected, dimension):
    shape = make_shape(nsite, nelec, nmax)
    assert isinstance(shape, tuple)
    assert all(isinstance(x, (int, np.integer)) for x in shape)
    assert shape == expected
    assert np.prod(shape) == dimension


@pytest.mark.parametrize(
    "nsite, nelec, nmax",
    [(0, (0, 0), 1), (64, (1, 0), 1), (4.5, (1, 0), 1),
     (4, 1, 2), (4, (1,), 2), (4, (-1, 0), 2),
     (4, (5, 0), 2), (4, (1.5, 0), 2),
     (4, (1, 0), -1), (4, (1, 0), 1.5)],
)
def test_invalid_parameters(nsite, nelec, nmax):
    with pytest.raises(ValueError):
        make_shape(nsite, nelec, nmax)
