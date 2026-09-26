"""Physical checkpoints for the worst-alpha MR-LF rank convergence study."""

import csv

import numpy as np
import pytest

from scripts.fig2b_mr_convergence import run_worst_convergence


def test_worst_alpha_and_frame_orbit_convergence(tmp_path):
    output = tmp_path / "mr_worst.csv"
    records = run_worst_convergence(
        theta_pool=(1.5,), nrandom=0, csv_path=output,
    )

    assert len(records) == 4
    assert all(record["alpha"] == pytest.approx(2.4) for record in records)
    assert [record["n_frames"] for record in records] == [1, 4, 8, 12]
    assert [record["rank"] for record in records] == [4, 16, 20, 36]
    np.testing.assert_allclose(
        [record["energy"] for record in records],
        [-2.93387001870493, -3.03667786336,
         -3.03816078627, -3.04036616703],
        atol=1e-9, rtol=0,
    )
    assert records[-1]["selected_theta"] == pytest.approx(1.5)
    assert all(record["gap_exact"] > 0 for record in records)
    assert records[-1]["gap_exact"] < records[0]["gap_exact"]
    assert records[-1]["min_retained_overlap"] > 1e-3

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[-1]["selected_theta"] == "1.5"
    assert rows[-1]["phase"] == "greedy_candidate_orbit"
