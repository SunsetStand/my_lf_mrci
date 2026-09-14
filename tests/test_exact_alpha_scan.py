"""Fast tests for exact alpha scanning, CSV persistence, and plotting."""

import csv

import numpy as np

from scripts import exact_alpha_scan as scan
from scripts.plot_fig2b_exact import plot_exact, read_exact_csv


def test_scan_records_and_writes_complete_csv(monkeypatch, tmp_path):
    def fake_kernel(tmat, interaction, coupling, hpp, nsite, nelec, nmax):
        energy = -2.0 - coupling**2
        return energy, None, 1e-9

    monkeypatch.setattr(scan.myep, "kernel", fake_kernel)
    csv_path = tmp_path / "data" / "fig2b_exact.csv"

    records = scan.run_exact_scan(
        np.array([0.0, 0.2, 0.4]),
        nmax=2,
        csv_path=csv_path,
    )

    assert len(records) == 3
    assert records[0][2:4] == (2, 324)
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 3
    assert tuple(rows[0]) == scan.CSV_COLUMNS
    assert rows[1]["alpha"] == "0.2"
    assert rows[1]["Nmax"] == "2"
    assert rows[1]["D"] == "324"
    assert rows[1]["convention"] == "paper_uncentered"


def test_plot_reads_csv_and_creates_png_and_pdf(monkeypatch, tmp_path):
    def fake_kernel(tmat, interaction, coupling, hpp, nsite, nelec, nmax):
        return -2.0 - coupling**2, None, 1e-9

    monkeypatch.setattr(scan.myep, "kernel", fake_kernel)
    csv_path = tmp_path / "fig2b_exact.csv"
    scan.run_exact_scan(np.array([0.4, 0.0, 0.2]), nmax=1, csv_path=csv_path)

    alpha, energy, metadata = read_exact_csv(csv_path)
    np.testing.assert_allclose(alpha, [0.0, 0.2, 0.4])
    np.testing.assert_allclose(energy, [-2.0, -2.1, -2.2])
    assert metadata["omega"] == "0.5"

    png_path = tmp_path / "figures" / "exact.png"
    pdf_path = tmp_path / "figures" / "exact.pdf"
    returned_paths = plot_exact(csv_path, png_path, pdf_path)

    assert returned_paths == (png_path, pdf_path)
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0
