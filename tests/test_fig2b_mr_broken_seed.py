"""Physical and output checks for the pre-break cross-alpha MR-LF scan."""

import csv

import numpy as np
import pytest

from scripts.fig2b_mr_broken_seed_scan import (
    DEFAULT_CSV_PATH, run_broken_seed_scan,
)
from scripts.plot_fig2b_exact import read_exact_csv
from scripts.plot_fig2b_mr import read_mr_csv
from scripts.plot_fig2b_mr_broken_seed import (
    plot_broken_seed, read_broken_seed_csv,
)
from src import lf_mr, my_direct_ep


def test_two_targets_and_frame_archive_reproduce_alpha_2p2(tmp_path):
    csv_path = tmp_path / "scan.csv"
    frames_path = tmp_path / "frames.npz"
    rows = run_broken_seed_scan(
        np.array([0.0, 2.2]), nrandom=0,
        csv_path=csv_path, frames_path=frames_path,
    )
    assert len(rows) == 2
    assert rows[0]["energy_augmented"] == pytest.approx(-2.0, abs=1e-12)
    assert rows[1]["energy_base"] == pytest.approx(-2.846976272803, abs=1e-9)
    assert rows[1]["energy_augmented"] == pytest.approx(
        -2.930809431151, abs=1e-9
    )
    assert rows[1]["rank_base"] == 4
    assert rows[1]["rank_augmented"] == 20
    assert rows[1]["energy_gain"] > 0.08
    assert rows[1]["norm_error"] < 1e-10
    assert rows[1]["density_imbalance"] < 1e-10
    assert rows[1]["min_retained_overlap"] > 0.28
    assert rows[1]["energy_cut_loose"] == pytest.approx(
        rows[1]["energy_augmented"], abs=1e-10
    )
    assert rows[1]["energy_cut_tight"] == pytest.approx(
        rows[1]["energy_augmented"], abs=1e-10
    )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    assert [float(row["alpha"]) for row in saved] == [0.0, 2.2]
    with np.load(frames_path) as frames:
        np.testing.assert_array_equal(frames["target_alpha"], [0.0, 2.2])
        assert frames["source_lam"].shape == (4, 4)
        assert frames["target_best_lam"].shape == (2, 4, 4)
        assert frames["target_cs_shift"].shape == (2, 4)
        orbits = [
            lf_mr.lf_translation_orbit(frames["target_best_lam"][-1],
                                       frames["target_best_shift"][-1]),
            lf_mr.lf_translation_orbit(frames["target_cs_lam"][-1],
                                       frames["target_cs_shift"][-1]),
            lf_mr.lf_translation_orbit(frames["source_lam"],
                                       frames["source_shift"]),
        ]
    lam = np.concatenate([orbit[0] for orbit in orbits])
    shift = np.concatenate([orbit[1] for orbit in orbits])
    tmat = my_direct_ep.electron_ring_hopping(4, -1.0)
    g = my_direct_ep.alpha_to_g(2.2, 0.5)
    hmat = lf_mr.lf_frame_hamiltonian(tmat, g, 0.5, lam, shift)
    smat = lf_mr.lf_frame_overlap(lam, shift)
    reproduced, _, rank = lf_mr.lf_noci_lowest(hmat, smat)
    assert rank == 20
    assert reproduced == pytest.approx(rows[1]["energy_augmented"], abs=1e-10)


def test_full_saved_scan_improves_gap_and_creates_separate_plot(tmp_path):
    new, _ = read_broken_seed_csv(DEFAULT_CSV_PATH)
    old, _ = read_mr_csv()
    exact_alpha, exact_energy, _ = read_exact_csv(
        DEFAULT_CSV_PATH.parent / "fig2b_exact.csv"
    )
    assert len(new["alpha"]) == 12
    np.testing.assert_allclose(new["alpha"], exact_alpha[:12], atol=1e-12)
    np.testing.assert_allclose(
        new["energy_base"], old["mr_energy"][:12], atol=1e-8
    )
    np.testing.assert_array_equal(new["rank_augmented"], 20)
    new_gap = new["energy_augmented"][1:] - exact_energy[1:12]
    old_gap = old["mr_energy"][1:12] - exact_energy[1:12]
    assert np.all(new_gap > 0)
    assert np.all(new_gap < old_gap)

    with DEFAULT_CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(int(row["rank_cut_loose"]) == 20 for row in rows)
    assert all(int(row["rank_cut_tight"]) == 20 for row in rows)
    assert all(float(row["min_retained_overlap"]) > 0.28 for row in rows)
    assert all(
        abs(float(row["energy_augmented"]) - float(row["energy_cut_loose"])) < 1e-10
        and abs(float(row["energy_augmented"]) - float(row["energy_cut_tight"])) < 1e-10
        for row in rows
    )

    png = tmp_path / "new.png"
    pdf = tmp_path / "new.pdf"
    assert plot_broken_seed(png_path=png, pdf_path=pdf) == (png, pdf)
    assert png.stat().st_size > 0
    assert pdf.stat().st_size > 0
