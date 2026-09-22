"""Acceptance tests for one full-matrix LF-HF coupling point."""

import numpy as np
import pytest

from src.lf_mp import lf_hf_full_alpha_point
from src.my_direct_ep import electron_ring_hopping


@pytest.mark.parametrize(
    ("alpha", "expected_energy", "expected_sorted_density"),
    [
        (0.0, -2.0, [0.25, 0.25, 0.25, 0.25]),
        (2.4, -2.93387001870493, [0.04401367, 0.10901334, 0.10901334, 0.73795966]),
    ],
)
def test_reports_energy_density_and_convergence_diagnostics(
    alpha: float,
    expected_energy: float,
    expected_sorted_density: list[float],
) -> None:
    omega = 0.5
    best, results = lf_hf_full_alpha_point(
        alpha,
        electron_ring_hopping(4, -1.0),
        omega,
        nrandom=0,
    )

    assert any(best is result for result in results)
    assert best.success
    assert best.alpha == alpha
    assert best.g == pytest.approx(np.sqrt(alpha * omega), abs=0.0)
    assert best.fun == pytest.approx(expected_energy, abs=1e-11)
    np.testing.assert_allclose(best.density, np.abs(best.coeff) ** 2, atol=1e-14)
    np.testing.assert_allclose(np.sum(best.density), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.sort(best.density), expected_sorted_density, atol=1e-8)
    assert best.density_imbalance == pytest.approx(
        max(expected_sorted_density) - min(expected_sorted_density),
        abs=1e-8,
    )
    assert len(results) == 5
    assert best.nstart == 5
    assert best.nconverged == sum(bool(result.success) for result in results)


@pytest.mark.parametrize("alpha", [-0.1, np.nan, np.inf, 1.0j])
def test_rejects_invalid_alpha(alpha: object) -> None:
    with pytest.raises(ValueError):
        lf_hf_full_alpha_point(alpha, np.zeros((4, 4)), omega=0.5)


@pytest.mark.parametrize("omega", [0.0, -0.5, np.nan, np.inf, 1.0j])
def test_rejects_invalid_frequency(omega: object) -> None:
    with pytest.raises(ValueError):
        lf_hf_full_alpha_point(0.8, np.zeros((4, 4)), omega)


@pytest.mark.parametrize("tmat", [np.zeros(4), np.zeros((3, 4))])
def test_rejects_invalid_hopping_shape(tmat: np.ndarray) -> None:
    with pytest.raises(ValueError):
        lf_hf_full_alpha_point(0.8, tmat, omega=0.5)
