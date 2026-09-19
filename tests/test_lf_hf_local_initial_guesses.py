"""Acceptance tests for reproducible local LF-HF starting points."""

import numpy as np
import pytest

from src.lf_mp import lf_hf_local_initial_guesses, lf_hf_local_multistart
from src.my_direct_ep import electron_ring_hopping


def test_first_row_is_exact_cs_point_with_expected_shape_and_dtype():
    starts = lf_hf_local_initial_guesses(
        norb=4,
        g=0.4,
        omega=0.5,
        nrandom=3,
        seed=19,
    )

    assert starts.shape == (4, 8)
    assert starts.dtype == np.float64
    np.testing.assert_array_equal(starts[0, :4], np.zeros(4))
    np.testing.assert_array_equal(starts[0, 4:], np.full(4, -0.2))


def test_seed_is_reproducible_and_does_not_use_global_random_state():
    state = np.random.get_state()
    try:
        np.random.seed(37)
        expected_global_values = np.random.random(3)
        np.random.seed(37)

        first = lf_hf_local_initial_guesses(4, 0.4, 0.5, seed=19)
        actual_global_values = np.random.random(3)
        second = lf_hf_local_initial_guesses(4, 0.4, 0.5, seed=19)
        different = lf_hf_local_initial_guesses(4, 0.4, 0.5, seed=20)
    finally:
        np.random.set_state(state)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[1:], different[1:])
    np.testing.assert_array_equal(actual_global_values, expected_global_values)


def test_zero_random_starts_returns_only_cs_point():
    starts = lf_hf_local_initial_guesses(4, 0.4, 0.5, nrandom=0)

    assert starts.shape == (1, 8)
    np.testing.assert_array_equal(starts[0], np.r_[np.zeros(4), np.full(4, -0.2)])


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("norb", 0),
        ("norb", 4.0),
        ("nrandom", -1),
        ("nrandom", 2.5),
        ("g", np.nan),
        ("g", 1.0j),
        ("omega", 0.0),
        ("omega", np.inf),
        ("random_scale", 0.0),
        ("random_scale", np.inf),
    ],
)
def test_rejects_invalid_inputs(keyword, value):
    arguments = {"norb": 4, "g": 0.4, "omega": 0.5}
    arguments[keyword] = value

    with pytest.raises(ValueError):
        lf_hf_local_initial_guesses(**arguments)


@pytest.mark.parametrize(
    ("alpha", "expected_energy", "expected_max_density"),
    [
        (2.2, -2.82487018210281, 0.25),
        (2.4, -2.924947166702873, 0.77026095),
    ],
)
def test_default_starts_distinguish_uniform_and_broken_solutions(
    alpha,
    expected_energy,
    expected_max_density,
):
    hopping = electron_ring_hopping(4, -1.0)
    omega = 0.5
    g = np.sqrt(alpha * omega)
    starts = lf_hf_local_initial_guesses(4, g, omega)

    best, results = lf_hf_local_multistart(starts, hopping, g, omega)

    assert any(result.success for result in results)
    np.testing.assert_allclose(best.fun, expected_energy, atol=1e-11)
    np.testing.assert_allclose(np.max(np.abs(best.coeff) ** 2), expected_max_density, atol=1e-8)
