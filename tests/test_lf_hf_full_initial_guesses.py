"""Acceptance tests for reproducible full-matrix LF-HF starting points."""

import numpy as np
import pytest

from src.lf_mp import lf_hf_full_initial_guesses, lf_hf_full_optimize


def _ring_hopping(norb: int) -> np.ndarray:
    hopping = np.zeros((norb, norb))
    for p in range(norb):
        hopping[p, (p + 1) % norb] = -1.0
        hopping[(p + 1) % norb, p] = -1.0
    return hopping


def test_deterministic_starts_have_fixed_gauge_layout() -> None:
    norb = 4
    omega = 0.5
    g = np.sqrt(2.4 * omega)

    guesses = lf_hf_full_initial_guesses(norb, g, omega, nrandom=0)

    assert guesses.shape == (1 + norb, norb, norb)
    assert guesses.dtype == np.float64
    np.testing.assert_allclose(guesses[0], g / (omega * norb), atol=0.0)
    for site in range(norb):
        expected = np.zeros((norb, norb))
        expected[site, site] = g / omega
        np.testing.assert_allclose(guesses[1 + site], expected, atol=0.0)


def test_uniform_random_starts_are_reproducible_and_rng_local() -> None:
    kwargs = {"norb": 4, "g": 0.7, "omega": 0.5, "nrandom": 3, "random_scale": 0.4}
    saved_state = np.random.get_state()
    try:
        np.random.seed(314159)
        expected_global_draw = np.random.random(4)
        np.random.seed(314159)
        first = lf_hf_full_initial_guesses(**kwargs, seed=11)
        actual_global_draw = np.random.random(4)
    finally:
        np.random.set_state(saved_state)

    second = lf_hf_full_initial_guesses(**kwargs, seed=11)
    different_seed = lf_hf_full_initial_guesses(**kwargs, seed=12)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:5], different_seed[:5])
    assert not np.array_equal(first[5:], different_seed[5:])
    np.testing.assert_array_equal(actual_global_draw, expected_global_draw)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("norb", 0),
        ("norb", True),
        ("nrandom", -1),
        ("nrandom", False),
        ("g", np.nan),
        ("g", 1j),
        ("omega", 0.0),
        ("omega", np.inf),
        ("random_scale", 0.0),
        ("random_scale", np.nan),
    ],
)
def test_rejects_invalid_inputs(keyword: str, value: object) -> None:
    kwargs = {"norb": 4, "g": 0.7, "omega": 0.5, "nrandom": 2, "random_scale": 0.4}
    kwargs[keyword] = value
    with pytest.raises(ValueError):
        lf_hf_full_initial_guesses(**kwargs)


def test_localized_start_reaches_paper_alpha_2p4_broken_solution() -> None:
    norb = 4
    omega = 0.5
    g = np.sqrt(2.4 * omega)
    guesses = lf_hf_full_initial_guesses(norb, g, omega, nrandom=0)

    result = lf_hf_full_optimize(guesses[1], _ring_hopping(norb), g, omega)

    assert result.success
    assert result.fun == pytest.approx(-2.93387001870493, abs=1e-11)
    np.testing.assert_allclose(
        np.abs(result.coeff) ** 2,
        [0.73795966, 0.10901334, 0.04401367, 0.10901334],
        atol=1e-8,
    )
    assert result.shift_residual < 1e-8
