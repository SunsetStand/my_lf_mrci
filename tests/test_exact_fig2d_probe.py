"""Fast tests for Fig. 2d probe metadata and convergence labels."""

import csv
from itertools import pairwise

from scripts import exact_fig2d_probe as probe


def test_probe_grid_csv_and_convergence_flags(monkeypatch, tmp_path):
    energies = {
        0: -2.0,
        2: -2.1,
        4: -2.1000005,
        6: -2.1000006,
    }

    def fake_kernel(*args, **kwargs):
        nmax = args[6]
        return energies[nmax], None, 1e-10

    monkeypatch.setattr(probe.myep, "kernel", fake_kernel)
    csv_path = tmp_path / "fig2d_exact_probe.csv"
    records = probe.run_fig2d_probe(
        alpha_values=(0.0, 2.0),
        omega_values=(0.5,),
        nmax_values=(2, 4, 6),
        csv_path=csv_path,
    )

    assert [record["Nmax"] for record in records] == [0, 2, 4, 6]
    assert records[0]["cutoff_converged"] is True
    assert records[-1]["cutoff_converged"] is True
    assert records[-1]["residual_ok"] is True
    assert records[-1]["convention"] == "paper_uncentered_probe"

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert tuple(rows[0]) == probe.CSV_COLUMNS
    assert rows[0]["nelec"] == "(2, 2)"
    assert rows[-1]["cutoff_converged"] == "True"


def test_bad_residual_prevents_cutoff_convergence(monkeypatch):
    def fake_kernel(*args, **kwargs):
        nmax = args[6]
        return -2.0 - 1e-8 * nmax, None, 2e-8

    monkeypatch.setattr(probe.myep, "kernel", fake_kernel)
    records = probe.run_fig2d_probe(
        alpha_values=(2.0,),
        omega_values=(5.0,),
        nmax_values=(2, 4, 6),
        csv_path=None,
    )

    assert all(record["residual_ok"] is False for record in records)
    assert all(record["cutoff_converged"] is False for record in records)


def test_residual_qualified_energies_are_variationally_nonincreasing(monkeypatch):
    energies = {2: -3.0, 4: -3.2, 6: -3.21, 8: -3.211}

    def fake_kernel(*args, **kwargs):
        return energies[args[6]], None, 1e-10

    monkeypatch.setattr(probe.myep, "kernel", fake_kernel)
    records = probe.run_fig2d_probe(
        alpha_values=(4.0,),
        omega_values=(0.5,),
        nmax_values=(2, 4, 6, 8),
        csv_path=None,
    )
    qualified_energies = [
        float(record["energy"]) for record in records if record["residual_ok"]
    ]

    assert all(
        current <= previous + 1e-12
        for previous, current in pairwise(qualified_energies)
    )
