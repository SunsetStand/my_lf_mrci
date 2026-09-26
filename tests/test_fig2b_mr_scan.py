"""Physical checkpoints for the fixed-recipe Fig. 2b MR-LF scan."""

import csv

import numpy as np
import pytest

from scripts.fig2b_mr_scan import FRAME_RECIPE, run_mr_scan


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
