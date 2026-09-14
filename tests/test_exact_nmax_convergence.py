"""Fast tests for automatic phonon-cutoff convergence logic."""

import numpy as np

from scripts import exact_nmax_convergence as convergence


def test_stops_after_two_consecutive_small_energy_changes(monkeypatch):
    energies = {
        0: -2.0,
        2: -2.1,
        4: -2.1000005,
        6: -2.1000006,
        8: -2.10000061,
    }
    calls = []

    def fake_kernel(tmat, interaction, coupling, hpp, nsite, nelec, nmax):
        calls.append((nmax, coupling))
        return energies[nmax], None, 1e-8

    monkeypatch.setattr(convergence.myep, "kernel", fake_kernel)

    records, converged = convergence.run_convergence(
        alpha=3.0,
        nmax_values=range(0, 10, 2),
        cutoff_tolerance=1e-6,
        required_consecutive=2,
    )

    assert converged is True
    assert [record[0] for record in records] == [0, 2, 4, 6]
    assert [record[1] for record in records] == [4, 324, 2500, 9604]
    assert records[0][4] is None
    np.testing.assert_allclose(records[-1][4], 1e-7, atol=1e-14)
    assert [nmax for nmax, _ in calls] == [0, 2, 4, 6]
    np.testing.assert_allclose(calls[0][1], np.sqrt(1.5), atol=1e-14)


def test_large_change_resets_consecutive_counter(monkeypatch):
    energies = {
        0: -2.0,
        2: -2.0000005,
        4: -2.01,
        6: -2.0100005,
        8: -2.0100006,
        10: -2.01000061,
    }
    calls = []

    def fake_kernel(tmat, interaction, coupling, hpp, nsite, nelec, nmax):
        calls.append(nmax)
        return energies[nmax], None, 1e-8

    monkeypatch.setattr(convergence.myep, "kernel", fake_kernel)

    records, converged = convergence.run_convergence(
        alpha=1.0,
        nmax_values=range(0, 12, 2),
        cutoff_tolerance=1e-6,
        required_consecutive=2,
    )

    assert converged is True
    assert calls == [0, 2, 4, 6, 8]
    assert records[-1][0] == 8
