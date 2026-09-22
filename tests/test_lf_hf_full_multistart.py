"""Acceptance tests for full-matrix LF-HF multistart selection."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from src import lf_mp
from src.my_direct_ep import electron_ring_hopping


def test_selects_paper_broken_solution_and_preserves_starts() -> None:
    hopping = electron_ring_hopping(4, -1.0)
    omega = 0.5
    g = np.sqrt(2.4 * omega)
    starts = lf_mp.lf_hf_full_initial_guesses(4, g, omega, nrandom=0)
    original_starts = starts.copy()

    best, results = lf_mp.lf_hf_full_multistart(starts, hopping, g, omega)

    assert len(results) == 5
    assert all(result.success for result in results)
    assert any(best is result for result in results)
    assert results[0].fun == pytest.approx(-2.92593345169089, abs=1e-11)
    assert best.fun == pytest.approx(-2.93387001870493, abs=1e-11)
    np.testing.assert_allclose(
        np.sort(np.abs(best.coeff) ** 2),
        [0.04401367, 0.10901334, 0.10901334, 0.73795966],
        atol=1e-8,
    )
    np.testing.assert_array_equal(starts, original_starts)


def test_ignores_failed_result_even_when_its_energy_is_lower(monkeypatch) -> None:
    outcomes = iter(
        [
            OptimizeResult(success=False, fun=-10.0),
            OptimizeResult(success=True, fun=-2.0),
        ]
    )

    def fake_optimize(*args, **kwargs):
        return next(outcomes)

    monkeypatch.setattr(lf_mp, "lf_hf_full_optimize", fake_optimize)
    best, results = lf_mp.lf_hf_full_multistart(
        np.zeros((2, 4, 4)),
        np.zeros((4, 4)),
        g=0.4,
        omega=0.5,
    )

    assert len(results) == 2
    assert best is results[1]


def test_raises_when_all_optimizations_fail(monkeypatch) -> None:
    def fake_optimize(*args, **kwargs):
        return OptimizeResult(success=False, fun=-2.0)

    monkeypatch.setattr(lf_mp, "lf_hf_full_optimize", fake_optimize)
    with pytest.raises(RuntimeError, match="failed"):
        lf_mp.lf_hf_full_multistart(
            np.zeros((2, 4, 4)),
            np.zeros((4, 4)),
            g=0.4,
            omega=0.5,
        )


@pytest.mark.parametrize(
    ("mu0s", "tmat"),
    [
        (np.zeros((1, 4, 4)), np.zeros(4)),
        (np.zeros((4, 4)), np.zeros((4, 4))),
        (np.zeros((0, 4, 4)), np.zeros((4, 4))),
        (np.zeros((1, 4, 3)), np.zeros((4, 4))),
        (np.full((1, 4, 4), np.nan), np.zeros((4, 4))),
        (np.full((1, 4, 4), 1.0j), np.zeros((4, 4))),
    ],
)
def test_rejects_invalid_hopping_or_start_arrays(mu0s: np.ndarray, tmat: np.ndarray) -> None:
    with pytest.raises(ValueError):
        lf_mp.lf_hf_full_multistart(mu0s, tmat, g=0.4, omega=0.5)
