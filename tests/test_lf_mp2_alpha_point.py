"""Acceptance tests for one optimized full-matrix LF-HF/LF-MP2 point."""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

from src import lf_mp
from src.my_direct_ep import electron_ring_hopping

MP2_FIELDS = {
    "hf_energy",
    "mp2_correction",
    "total_energy",
    "max_total",
    "mo_energy",
    "mo_coeff",
    "occupations",
    "total_occupations",
    "pure_matrix_elements",
    "single_matrix_elements",
    "pure_denominators",
    "single_denominators",
}


def test_zero_coupling_optimizes_reference_and_adds_mp2_fields():
    tmat = electron_ring_hopping(4, -1.0)

    best, results = lf_mp.lf_mp2_alpha_point(
        0.0,
        tmat,
        omega=0.5,
        max_total=3,
        nrandom=0,
    )

    assert any(best is result for result in results)
    assert best.success
    assert MP2_FIELDS <= set(best)
    assert best.alpha == 0.0
    assert best.g == 0.0
    assert best.max_total == 3
    assert best.fun == pytest.approx(-2.0, abs=1e-14)
    assert best.hf_energy == pytest.approx(best.fun, abs=1e-14)
    assert best.mp2_correction == 0.0
    assert best.total_energy == pytest.approx(-2.0, abs=1e-14)
    np.testing.assert_allclose(
        best.mo_energy,
        [-2.0, 0.0, 0.0, 2.0],
        atol=1e-14,
    )
    assert best.occupations.shape == (34, 4)
    assert best.single_matrix_elements.shape == (34, 1, 3)


def test_passes_selected_reference_and_options_through_both_stages(monkeypatch):
    tmat = np.array([[0.0, -1.0], [-1.0, 0.0]])
    lam = np.array([[0.2, -0.1], [0.05, 0.3]])
    shift = np.zeros(2)
    best = OptimizeResult(
        success=True,
        fun=-1.2,
        g=0.4,
        lam=lam,
        shift=shift,
        coeff=np.array([1.0, 0.0]),
    )
    results = [OptimizeResult(success=False, fun=-2.0), best]
    calls = {}

    def fake_hf(alpha, passed_tmat, passed_omega, **kwargs):
        calls["hf"] = (alpha, passed_tmat, passed_omega, kwargs)
        return best, results

    def fake_reference(
        passed_tmat,
        passed_g,
        passed_omega,
        passed_shift,
        passed_lam,
        *,
        max_total,
    ):
        calls["reference"] = (
            passed_tmat,
            passed_g,
            passed_omega,
            passed_shift,
            passed_lam,
            max_total,
        )
        return {
            "hf_energy": -1.2,
            "mp2_correction": -0.1,
            "total_energy": -1.3,
            "max_total": max_total,
        }

    monkeypatch.setattr(lf_mp, "lf_hf_full_alpha_point", fake_hf)
    monkeypatch.setattr(lf_mp, "lf_mp2_reference_point", fake_reference)

    returned_best, returned_results = lf_mp.lf_mp2_alpha_point(
        0.32,
        tmat,
        omega=0.5,
        max_total=7,
        nrandom=3,
        random_scale=0.2,
        seed=11,
        gtol=2e-8,
        max_cycle=123,
    )

    assert returned_best is best
    assert returned_results is results
    assert calls["hf"][0] == 0.32
    assert calls["hf"][1] is tmat
    assert calls["hf"][2] == 0.5
    assert calls["hf"][3] == {
        "nrandom": 3,
        "random_scale": 0.2,
        "seed": 11,
        "gtol": 2e-8,
        "max_cycle": 123,
    }
    assert calls["reference"][0] is tmat
    assert calls["reference"][1] == 0.4
    assert calls["reference"][2] == 0.5
    assert calls["reference"][3] is shift
    assert calls["reference"][4] is lam
    assert calls["reference"][5] == 7
    assert best.hf_energy == -1.2
    assert best.mp2_correction == -0.1
    assert best.total_energy == -1.3


@pytest.mark.parametrize(
    ("max_total", "exception"),
    [(0, ValueError), (-1, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_rejects_invalid_cutoff_before_starting_hf(
    monkeypatch,
    max_total,
    exception,
):
    def unexpected_hf(*args, **kwargs):
        raise AssertionError("LF-HF must not start for an invalid cutoff")

    monkeypatch.setattr(lf_mp, "lf_hf_full_alpha_point", unexpected_hf)

    with pytest.raises(exception):
        lf_mp.lf_mp2_alpha_point(
            0.5,
            np.zeros((2, 2)),
            omega=0.5,
            max_total=max_total,
        )
