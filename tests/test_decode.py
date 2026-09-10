"""Acceptance tests for phonon address decoding."""

import numpy as np
import pytest

from src.my_direct_ep import decode, encode


@pytest.mark.parametrize("nsite, d", [(4, 3), (3, 2), (4, 1)])
def test_decode_all_addresses(nsite, d):
    for address in range(d**nsite):
        occupations = decode(address, nsite, d)
        assert isinstance(occupations, tuple)
        assert len(occupations) == nsite
        assert all(isinstance(m, (int, np.integer)) for m in occupations)
        assert occupations == tuple(
            np.unravel_index(address, (d,) * nsite, order="C")
        )
        assert encode(occupations, d) == address


@pytest.mark.parametrize(
    "address, nsite, d, expected",
    [(34, 4, 3, (1, 0, 2, 1)), (1, 4, 3, (0, 0, 0, 1)),
     (0, 4, 1, (0, 0, 0, 0))],
)
def test_decode_examples(address, nsite, d, expected):
    assert decode(address, nsite, d) == expected


@pytest.mark.parametrize(
    "address, nsite, d",
    [(-1, 4, 3), (81, 4, 3), (0.5, 4, 3), (0, 0, 3),
     (0, 4, 0), (0, 2.5, 3), (0, 4, 2.5)],
)
def test_decode_rejects_invalid_parameters(address, nsite, d):
    with pytest.raises(ValueError):
        decode(address, nsite, d)
