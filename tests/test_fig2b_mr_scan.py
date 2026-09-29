"""Physical checkpoints for the fixed-recipe Fig. 2b MR-LF scan."""

import csv
from pathlib import Path

import numpy as np
import pytest

from scripts.fig2b_mr_scan import FRAME_RECIPE, run_mr_scan
from src import lf_mp, lf_mr, my_direct_ep


def test_two_point_scan_records_variational_energy_and_symmetry(tmp_path):
    output = tmp_path / "fig2b_mr.csv"
    records = run_mr_scan(
        np.array([0.0, 2.4]), nrandom=0, csv_path=output,
    )
    assert len(records) == 2
    zero, middle = records
    assert zero["mr_energy"] == pytest.approx(-2.0, abs=1e-12)
    assert zero["mr_rank"] == 4
    assert middle["lf_energy"] == pytest.approx(-2.93387001870493, abs=1e-10)
    assert middle["mr_orbit_energy"] == pytest.approx(-3.03667786336, abs=1e-9)
    assert middle["mr_energy"] == pytest.approx(-3.03816078627, abs=1e-9)
    assert middle["mr_orbit_rank"] == 16
    assert middle["mr_rank"] == 20
    assert middle["mr_energy"] < middle["mr_orbit_energy"]
    assert middle["mr_orbit_energy"] < middle["lf_energy"]
    for record in records:
        np.testing.assert_allclose(record["mr_density"], 0.25, atol=1e-9)
        assert record["mr_norm_error"] < 1e-10

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert [float(row["alpha"]) for row in rows] == [0.0, 2.4]
    assert all(row["coupling_convention"] == "paper_uncentered" for row in rows)
    assert all(row["frame_recipe"] == FRAME_RECIPE for row in rows)


def test_alpha_2p4_broken_orbit_improves_alpha_2p2_mr_energy():
    omega = 0.5
    tmat = my_direct_ep.electron_ring_hopping(4, -1.0)
    settings = dict(nrandom=8, random_scale=0.5, seed=0, gtol=1e-8, max_cycle=500)
    best22, results22 = lf_mp.lf_hf_full_alpha_point(2.2, tmat, omega, **settings)
    best24, _ = lf_mp.lf_hf_full_alpha_point(2.4, tmat, omega, **settings)
    assert best22.density_imbalance < 1e-8
    assert best24.density_imbalance > 0.6

    best_lam, best_shift = lf_mr.lf_translation_orbit(best22.lam, best22.shift)
    cs_lam, cs_shift = lf_mr.lf_translation_orbit(
        results22[0].lam, results22[0].shift
    )
    base_lam = np.concatenate((best_lam, cs_lam))
    base_shift = np.concatenate((best_shift, cs_shift))
    g22 = my_direct_ep.alpha_to_g(2.2, omega)

    exact_path = Path(__file__).resolve().parents[1] / "data" / "fig2b_exact.csv"
    with exact_path.open(newline="", encoding="utf-8") as handle:
        exact_row = next(
            row for row in csv.DictReader(handle) if float(row["alpha"]) == 2.2
        )
    assert exact_row["convention"] == "paper_uncentered"
    assert int(exact_row["Nmax"]) == 24
    exact = float(exact_row["energy"])

    for overlap_cut in (1e-8, 1e-10, 1e-12):
        base, augmented, base_rank, augmented_rank = lf_mr.lf_noci_trial_orbit(
            tmat, g22, omega, base_lam, base_shift,
            best24.lam, best24.shift, overlap_cut=overlap_cut,
        )
        assert base_rank == 4
        assert augmented_rank == 20
        assert base == pytest.approx(-2.846976272803, abs=1e-9)
        assert augmented == pytest.approx(-2.930809431151, abs=1e-9)
        assert 0 < augmented - exact < 0.005
        assert augmented - exact < base - exact
