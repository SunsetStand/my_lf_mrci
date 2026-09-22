"""Tests for persistent combined Fig. 2b MP2 data and plotting."""

import csv

import numpy as np

from scripts import fig2b_mp2_scan as scan
from scripts.plot_fig2b_hf import read_hf_csv
from scripts.plot_fig2b_mp2 import plot_mp2, read_mp2_csv


def test_scan_writes_extended_csv_and_reader_sorts_rows(tmp_path):
    csv_path = tmp_path / "data" / "fig2b_mp2.csv"
    records = scan.run_mp2_scan(
        np.array([0.4, 0.0]),
        max_total=4,
        nrandom=0,
        csv_path=csv_path,
    )

    assert len(records) == 2
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert tuple(rows[0]) == scan.CSV_COLUMNS
    assert [row["alpha"] for row in rows] == ["0.4", "0"]
    assert rows[0]["lf_mp2_max_total"] == "4"
    assert rows[0]["lf_mp2_nconfig"] == "69"
    data, metadata = read_mp2_csv(csv_path)
    np.testing.assert_array_equal(data["alpha"], [0.0, 0.4])
    np.testing.assert_array_equal(data["lf_mp2_max_total"], [4, 4])
    np.testing.assert_allclose(
        data["lf_mp2_energy"],
        data["lf_energy"] + data["lf_mp2_correction"],
        atol=1e-14,
    )
    assert metadata["lf_ansatz"] == "full_density_diagonal"


def test_plot_creates_new_png_and_pdf_without_using_hf_output_paths(tmp_path):
    csv_path = tmp_path / "fig2b_mp2.csv"
    scan.run_mp2_scan(
        np.array([0.0, 0.4]),
        max_total=4,
        nrandom=0,
        csv_path=csv_path,
    )
    exact_csv_path = tmp_path / "fig2b_exact.csv"
    exact_csv_path.write_text(
        "alpha,energy,L,omega\n0,-2,4,0.5\n0.4,-2.15,4,0.5\n",
        encoding="utf-8",
    )
    png_path = tmp_path / "figures" / "fig2b_mp2.png"
    pdf_path = tmp_path / "figures" / "fig2b_mp2.pdf"

    returned = plot_mp2(csv_path, png_path, pdf_path, exact_csv_path)

    assert returned == (png_path, pdf_path)
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0
    assert not (tmp_path / "figures" / "fig2b_hf.png").exists()
    assert not (tmp_path / "figures" / "fig2b_hf.pdf").exists()


def test_persisted_mp2_scan_matches_existing_hf_columns_and_energy_identities():
    mp2_data, metadata = read_mp2_csv()
    hf_data, _ = read_hf_csv()

    assert metadata["lf_ansatz"] == "full_density_diagonal"
    np.testing.assert_allclose(mp2_data["alpha"], scan.DEFAULT_ALPHA_VALUES, atol=0.0)
    for key in (
        "cs_energy",
        "cs_mp2_correction",
        "cs_mp2_energy",
        "lf_energy",
        "cs_density_imbalance",
        "lf_density_imbalance",
    ):
        np.testing.assert_allclose(mp2_data[key], hf_data[key], atol=1e-11)
    np.testing.assert_array_equal(
        mp2_data["lf_mp2_max_total"], scan.DEFAULT_MAX_TOTAL
    )
    np.testing.assert_allclose(
        mp2_data["lf_mp2_correction"],
        mp2_data["lf_mp2_pure_correction"]
        + mp2_data["lf_mp2_single_correction"],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        mp2_data["lf_mp2_energy"],
        mp2_data["lf_energy"] + mp2_data["lf_mp2_correction"],
        atol=1e-14,
    )
    assert np.all(mp2_data["lf_mp2_correction"] <= 0.0)
