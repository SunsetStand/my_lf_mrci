"""Tests for persistent Fig. 2b HF data and plotting."""

import csv

import numpy as np
import pytest

from scripts import fig2b_hf_scan as scan
from scripts.plot_fig2b_hf import plot_hf, read_hf_csv


def test_default_grid_matches_existing_fig2b_exact_grid():
    expected = np.linspace(0.0, 3.0, 16)

    np.testing.assert_allclose(scan.DEFAULT_ALPHA_VALUES, expected, atol=0.0)


def test_persisted_scan_uses_full_lf_and_matches_paper_anchors():
    data, metadata = read_hf_csv()

    assert metadata["lf_ansatz"] == "full_density_diagonal"
    np.testing.assert_allclose(data["alpha"], scan.DEFAULT_ALPHA_VALUES, atol=0.0)
    anchors = {
        1.0: -2.380726824272466,
        2.2: -2.846976272802588,
        2.4: -2.933870018704931,
        2.8: -3.226922548557187,
        3.0: -3.390114000228852,
    }
    for alpha, expected_energy in anchors.items():
        index = int(np.flatnonzero(np.isclose(data["alpha"], alpha))[0])
        assert data["lf_energy"][index] == pytest.approx(expected_energy, abs=1e-12)

    index_2p2 = int(np.flatnonzero(np.isclose(data["alpha"], 2.2))[0])
    index_2p4 = int(np.flatnonzero(np.isclose(data["alpha"], 2.4))[0])
    assert data["lf_density_imbalance"][index_2p2] < 1e-8
    assert data["lf_density_imbalance"][index_2p4] > 0.69


def test_scan_writes_auditable_csv_and_reader_sorts_rows(tmp_path):
    csv_path = tmp_path / "data" / "fig2b_hf.csv"

    records = scan.run_hf_scan(
        np.array([0.4, 0.0]),
        nrandom=0,
        csv_path=csv_path,
    )

    assert len(records) == 2
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert tuple(rows[0]) == scan.CSV_COLUMNS
    assert [row["alpha"] for row in rows] == ["0.4", "0"]
    assert rows[0]["omega"] == "0.5"
    assert rows[0]["L"] == "4"
    assert rows[0]["t"] == "-1"
    assert rows[0]["coupling_convention"] == "paper_uncentered"
    assert rows[0]["cs_branch"] == "uniform_symmetric"
    assert rows[0]["lf_ansatz"] == "full_density_diagonal"
    assert rows[0]["lf_nstart"] == "5"
    assert rows[0]["nrandom"] == "0"

    data, metadata = read_hf_csv(csv_path)
    np.testing.assert_array_equal(data["alpha"], [0.0, 0.4])
    np.testing.assert_allclose(data["cs_energy"], [-2.0, -2.1], atol=1e-14)
    np.testing.assert_allclose(
        data["cs_mp2_correction"],
        [0.0, -23.0 * 0.4 / 180.0],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        data["cs_mp2_energy"],
        [-2.0, -2.0 - 17.0 * 0.4 / 45.0],
        atol=1e-14,
    )
    assert data["cs_density"].shape == (2, 4)
    assert data["lf_density"].shape == (2, 4)
    np.testing.assert_allclose(np.sum(data["lf_density"], axis=1), 1.0, atol=1e-14)
    assert metadata["L"] == "4"


def test_plot_creates_nonempty_png_and_pdf(tmp_path):
    csv_path = tmp_path / "fig2b_hf.csv"
    scan.run_hf_scan(np.array([0.0, 0.4]), nrandom=0, csv_path=csv_path)
    exact_csv_path = tmp_path / "fig2b_exact.csv"
    exact_csv_path.write_text(
        "alpha,energy,L,omega\n0,-2,4,0.5\n0.4,-2.15,4,0.5\n",
        encoding="utf-8",
    )
    png_path = tmp_path / "figures" / "fig2b_hf.png"
    pdf_path = tmp_path / "figures" / "fig2b_hf.pdf"

    returned = plot_hf(csv_path, png_path, pdf_path, exact_csv_path)

    assert returned == (png_path, pdf_path)
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0


def test_reader_rejects_empty_or_incomplete_csv(tmp_path):
    empty_path = tmp_path / "empty.csv"
    empty_path.write_text("alpha,cs_energy\n", encoding="utf-8")
    incomplete_path = tmp_path / "incomplete.csv"
    incomplete_path.write_text("alpha,cs_energy\n0,-2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no data"):
        read_hf_csv(empty_path)
    with pytest.raises(ValueError, match="missing columns"):
        read_hf_csv(incomplete_path)
