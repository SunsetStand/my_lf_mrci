"""Acceptance tests for the first assignment, including invalid inputs.

Run from the project root with python -m pytest.
These tests intentionally expose unfinished parts of the student submission.
"""

import numpy as np
import pytest

from src.my_direct_ep import alpha_to_g, electron_ring_hopping, encode


@pytest.mark.parametrize("alpha, omega, expected", [(0.0, 0.5, 0.0), (2.0, 0.5, 1.0)])
def test_alpha_to_g(alpha, omega, expected):
    assert alpha_to_g(alpha, omega) == pytest.approx(expected)


@pytest.mark.parametrize("alpha, omega", [(-1.0, 0.5), (1.0, 0.0)])
def test_alpha_rejects_invalid_parameters(alpha, omega):
    with pytest.raises(ValueError):
        alpha_to_g(alpha, omega)


def test_encode_all_81_configurations():
    addresses = []
    for occ in np.ndindex((3,) * 4):
        address = encode(occ, 3)
        assert isinstance(address, (int, np.integer))
        assert address == np.ravel_multi_index(occ, (3,) * 4, order="C")
        addresses.append(address)
    assert addresses == list(range(81))


def test_encode_vacuum_only():
    assert encode((0, 0, 0, 0), 1) == 0


@pytest.mark.parametrize(
    "occupations, d",
    [((-1, 0), 3), ((0, 3), 3), ((0, 0.5), 3),
     ((0, 0), 0), ((0, 0), -1), ((0, 0), 2.5)],
)
def test_encode_rejects_invalid_parameters(occupations, d):
    with pytest.raises(ValueError):
        encode(occupations, d)


@pytest.mark.parametrize("nsite, hopping", [(4, -1.0), (4, -0.5), (5, -1.0)])
def test_electron_ring(nsite, hopping):
    matrix = electron_ring_hopping(nsite, hopping)
    assert matrix.shape == (nsite, nsite)
    assert matrix.dtype == np.float64
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 0.0)
    for i in range(nsite):
        for j in range(nsite):
            distance = min(abs(i - j), nsite - abs(i - j))
            assert matrix[i, j] == (hopping if distance == 1 else 0.0)
    expected = np.sort(2 * hopping * np.cos(2 * np.pi * np.arange(nsite) / nsite))
    np.testing.assert_allclose(np.linalg.eigvalsh(matrix), expected, atol=1e-12)


@pytest.mark.parametrize("nsite", [2, 0])
def test_ring_rejects_small_lattice(nsite):
    with pytest.raises(ValueError):
        electron_ring_hopping(nsite)
