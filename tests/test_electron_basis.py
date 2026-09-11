"""Acceptance tests for electronic determinants and one-body links."""

import numpy as np
import pytest

from src.my_direct_ep import make_electron_basis


@pytest.mark.parametrize(
    "nelec, expected, shape",
    [(0, [0], (1, 0, 4)), (1, [1, 2, 4, 8], (4, 4, 4)),
     (2, [3, 5, 6, 9, 10, 12], (6, 6, 4)), (4, [15], (1, 4, 4))],
)
def test_electron_sectors(nelec, expected, shape):
    strings, links = make_electron_basis(4, nelec)
    assert strings.dtype == np.int64
    assert links.dtype == np.int32
    np.testing.assert_array_equal(strings, expected)
    assert links.shape == shape


def test_single_electron_links():
    strings, links = make_electron_basis(4, 1)
    for source, rows in enumerate(links):
        for a, i, target, sign in rows:
            assert strings[source] == 1 << i
            assert strings[target] == 1 << a
            assert sign == 1


def test_two_electron_negative_sign():
    strings, links = make_electron_basis(4, 2)
    source = int(np.flatnonzero(strings == 0b0101)[0])
    rows = links[source]
    selected = rows[(rows[:, 0] == 3) & (rows[:, 1] == 0)]
    assert selected.shape == (1, 4)
    assert strings[selected[0, 2]] == 0b1100
    assert selected[0, 3] == -1


@pytest.mark.parametrize(
    "nsite, nelec", [(0, 0), (64, 1), (4, -1), (4, 5), (4.5, 1), (4, 1.5)]
)
def test_invalid_parameters(nsite, nelec):
    with pytest.raises(ValueError):
        make_electron_basis(nsite, nelec)
