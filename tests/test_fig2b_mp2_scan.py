"""Tests for the combined Fig. 2b HF/MP2 scan."""

import math

import numpy as np
import pytest

from scripts import fig2b_mp2_scan as scan
from src import lf_mp
from src.my_direct_ep import electron_ring_hopping


def test_real_scan_adds_lf_mp2_and_preserves_hf_quantities():
    records = scan.run_mp2_scan(
        np.array([0.0, 2.4]),
        max_total=6,
        nrandom=0,
    )

    assert len(records) == 2
    zero, broken = records
    assert zero["lf_energy"] == pytest.approx(-2.0, abs=1e-14)
    assert zero["lf_mp2_correction"] == 0.0
    assert zero["lf_mp2_energy"] == pytest.approx(-2.0, abs=1e-14)
    assert broken["lf_energy"] == pytest.approx(-2.93387001870493, abs=1e-11)
    assert broken["lf_mp2_correction"] == pytest.approx(
        -0.02387913, abs=2e-8
    )
    for record in records:
        assert record["lf_mp2_max_total"] == 6
        assert record["lf_mp2_nconfig"] == math.comb(10, 4) - 1
        assert record["lf_mp2_correction"] == pytest.approx(
            record["lf_mp2_pure_correction"]
            + record["lf_mp2_single_correction"],
            abs=1e-14,
        )
        assert record["lf_mp2_energy"] == pytest.approx(
            record["lf_energy"] + record["lf_mp2_correction"], abs=1e-14
        )
        assert record["lf_mp2_correction"] <= 0.0


def test_default_cutoff_is_converged_against_sixteen_at_worst_anchor():
    tmat = electron_ring_hopping(4, -1.0)
    best, _ = lf_mp.lf_hf_full_alpha_point(2.2, tmat, omega=0.5, nrandom=0)

    default = lf_mp.lf_mp2_reference_point(
        tmat,
        best["g"],
        0.5,
        best["shift"],
        best["lam"],
        max_total=scan.DEFAULT_MAX_TOTAL,
    )
    reference = lf_mp.lf_mp2_reference_point(
        tmat,
        best["g"],
        0.5,
        best["shift"],
        best["lam"],
        max_total=16,
    )

    assert scan.DEFAULT_MAX_TOTAL == 10
    assert default["total_energy"] == pytest.approx(
        reference["total_energy"], abs=1e-12
    )


@pytest.mark.parametrize(
    "alpha_values",
    [
        np.array([]),
        np.zeros((1, 2)),
        np.array([-0.1]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([1.0j]),
    ],
)
def test_rejects_invalid_alpha_arrays(alpha_values):
    with pytest.raises(ValueError):
        scan.run_mp2_scan(alpha_values, nrandom=0)


@pytest.mark.parametrize(
    ("max_total", "exception"),
    [(0, ValueError), (1.5, TypeError), (True, TypeError)],
)
def test_rejects_invalid_total_phonon_cutoff(max_total, exception):
    with pytest.raises(exception):
        scan.run_mp2_scan(np.array([0.0]), max_total=max_total, nrandom=0)
