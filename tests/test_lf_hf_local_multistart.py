"""Acceptance tests for local-diagonal LF-HF multistart selection."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from src import lf_mp
from src.my_direct_ep import electron_ring_hopping


def test_selects_broken_symmetry_solution_and_preserves_input_order():
    hopping = electron_ring_hopping(4, -1.0)
    omega = 0.5
    alpha = 2.4
    g = np.sqrt(alpha * omega)
    symmetric = np.concatenate((np.zeros(4), np.full(4, -g / (4.0 * omega))))
    localized = symmetric.copy()
    localized[4] -= 0.5
    params0s = np.vstack((symmetric, localized))
    original_params = params0s.copy()

    best, results = lf_mp.lf_hf_local_multistart(params0s, hopping, g, omega)

    assert len(results) == 2
    assert all(result.success for result in results)
    assert best is results[1]
    np.testing.assert_allclose(results[0].fun, -2.9016212151555663, atol=1e-11)
    np.testing.assert_allclose(best.fun, -2.924947166702871, atol=1e-11)
    np.testing.assert_allclose(
        np.sort(np.abs(best.coeff) ** 2),
        np.sort([0.77026095, 0.09933737, 0.03106430, 0.09933737]),
        atol=1e-8,
    )
    np.testing.assert_array_equal(params0s, original_params)


def test_ignores_failed_result_even_when_its_energy_is_lower(monkeypatch):
    outcomes = iter(
        [
            OptimizeResult(success=False, fun=-10.0),
            OptimizeResult(success=True, fun=-2.0),
        ]
    )

    def fake_optimize(*args, **kwargs):
        return next(outcomes)

    monkeypatch.setattr(lf_mp, "lf_hf_local_optimize", fake_optimize)

    best, results = lf_mp.lf_hf_local_multistart(
        np.zeros((2, 8)),
        np.zeros((4, 4)),
        g=0.4,
        omega=0.5,
    )

    assert len(results) == 2
    assert best is results[1]


def test_raises_when_all_optimizations_fail(monkeypatch):
    def fake_optimize(*args, **kwargs):
        return OptimizeResult(success=False, fun=-2.0)

    monkeypatch.setattr(lf_mp, "lf_hf_local_optimize", fake_optimize)

    with pytest.raises(RuntimeError, match="failed"):
        lf_mp.lf_hf_local_multistart(
            np.zeros((2, 8)),
            np.zeros((4, 4)),
            g=0.4,
            omega=0.5,
        )


@pytest.mark.parametrize(
    ("params0s", "tmat"),
    [
        (np.zeros((1, 8)), np.zeros(4)),
        (np.zeros(8), np.zeros((4, 4))),
        (np.zeros((0, 8)), np.zeros((4, 4))),
        (np.zeros((1, 7)), np.zeros((4, 4))),
        (np.array([[0.0] * 7 + [np.nan]]), np.zeros((4, 4))),
        (np.array([[0.0] * 7 + [1.0j]]), np.zeros((4, 4))),
    ],
)
def test_rejects_invalid_hopping_or_start_arrays(params0s, tmat):
    with pytest.raises(ValueError):
        lf_mp.lf_hf_local_multistart(params0s, tmat, g=0.4, omega=0.5)
