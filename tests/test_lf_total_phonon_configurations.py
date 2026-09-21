"""Acceptance tests for the LF-MP total-phonon configuration table."""

import math

import numpy as np
import pytest

from src.lf_mp import lf_total_phonon_configurations


def test_two_modes_with_cutoff_two_has_expected_order_and_rows():
    configurations = lf_total_phonon_configurations(nmode=2, max_total=2)

    expected = np.array(
        [
            [0, 1],
            [1, 0],
            [0, 2],
            [1, 1],
            [2, 0],
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(configurations, expected)
    assert configurations.dtype == np.int64


def test_cutoff_is_on_collective_total_not_independent_mode_occupations():
    configurations = lf_total_phonon_configurations(nmode=2, max_total=2)
    rows = {tuple(row) for row in configurations}

    assert (2, 0) in rows
    assert (1, 1) in rows
    assert (2, 1) not in rows
    assert (2, 2) not in rows


def test_fig2b_size_matches_weak_composition_count():
    nmode = 4
    max_total = 16

    configurations = lf_total_phonon_configurations(nmode, max_total)
    totals = configurations.sum(axis=1)

    assert configurations.shape == (
        math.comb(max_total + nmode, nmode) - 1,
        nmode,
    )
    assert configurations.shape == (4844, 4)
    assert np.min(totals) == 1
    assert np.max(totals) == max_total
    assert np.all(totals[:-1] <= totals[1:])
    assert len({tuple(row) for row in configurations}) == len(configurations)


@pytest.mark.parametrize("name", ["nmode", "max_total"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_rejects_nonpositive_and_noninteger_arguments(name, value):
    arguments = {"nmode": 2, "max_total": 3}
    arguments[name] = value

    expected_exception = TypeError if isinstance(value, (float, bool)) else ValueError
    with pytest.raises(expected_exception):
        lf_total_phonon_configurations(**arguments)
