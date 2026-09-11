"""Acceptance tests for the phonon configuration table."""

import numpy as np
import pytest

from src.my_direct_ep import encode, phonon_configs


@pytest.mark.parametrize("nsite, nmax", [(2, 1), (4, 2), (4, 0)])
def test_configurations(nsite, nmax):
    d = nmax + 1
    configs = phonon_configs(nsite, nmax)
    assert configs.shape == (d**nsite, nsite)
    assert configs.dtype == np.int64
    expected = np.array(list(np.ndindex((d,) * nsite)), dtype=np.int64)
    np.testing.assert_array_equal(configs, expected)
    for address, row in enumerate(configs):
        assert encode(row, d) == address


def test_selected_rows():
    configs = phonon_configs(4, 2)
    np.testing.assert_array_equal(configs[34], [1, 0, 2, 1])
    np.testing.assert_array_equal(configs[-1], [2, 2, 2, 2])


@pytest.mark.parametrize("nsite, nmax", [(0, 2), (4, -1), (2.5, 2), (4, 1.5)])
def test_invalid_parameters(nsite, nmax):
    with pytest.raises(ValueError):
        phonon_configs(nsite, nmax)
