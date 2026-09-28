"""Physical validation of the centered Fig. 2d data driver."""
import numpy as np
import pytest

from scripts.fig2d_exact_scan import resize_guess, run_scan, solve_centered
from src import my_direct_ep as ep


@pytest.mark.parametrize("alpha,nmax", [(0., 0), (.8, 2)])
def test_driver_matches_existing_centered_kernel(alpha, nmax):
    omega = .5
    record, psi = solve_centered(alpha, omega, nmax)
    energy, _, residual = ep.kernel(
        ep.electron_ring_hopping(4), 4., np.sqrt(alpha*omega),
        omega*np.eye(4), 4, (2, 2), nmax,
        tol_residual=1e-8, max_cycle=500, lindep=1e-18,
        coupling_convention="centered",
    )
    assert record["energy_centered"] == pytest.approx(energy, abs=1e-10)
    assert record["solver_converged"]
    assert record["residual"] < 1e-8
    assert residual < 1e-8
    assert np.linalg.norm(psi) == pytest.approx(1.)


def test_real_cutoff_sequence_and_paper_energy_shift(tmp_path):
    records = run_scan([.4], [5.], [2, 4, 6, 8, 10], tmp_path/"scan.csv",
                       max_space=40)
    assert records[-1]["cutoff_converged"]
    assert len(records) >= 3
    assert all(r["delta_E"] < 1e-6 for r in records[-2:])
    for r in records:
        assert r["energy"] == pytest.approx(r["energy_centered"]-1.6)
        assert r["residual"] < 1e-8


def test_insufficient_cutoff_is_not_labeled_converged(tmp_path):
    records = run_scan([4.], [.5], [2, 4], tmp_path/"short.csv")
    assert not any(r["cutoff_converged"] for r in records)


def test_repeated_cutoffs_cannot_fake_convergence(tmp_path):
    with pytest.raises(ValueError, match="strictly increasing"):
        run_scan([.4], [5.], [2, 2, 2], tmp_path/"bad.csv")


def test_resize_does_not_change_input_or_mix_phonon_axes():
    original = np.zeros((6, 6, 3, 3, 3, 3))
    original[1, 4, 1, 2, 0, 1] = 1.
    before = original.copy()
    padded = resize_guess(original, 4)
    assert padded[1, 4, 1, 2, 0, 1] > .9
    assert not np.any(padded[:, :, 3:, :, :, :])
    np.testing.assert_array_equal(original, before)


def test_stalled_davidson_falls_back_without_relaxing_residual():
    record, _ = solve_centered(.8, .5, 2, max_cycle=1)
    assert record["solver"] == "eigsh_after_davidson"
    assert record["solver_converged"]
    assert record["residual"] < 1e-8
    assert record["energy_centered"] == pytest.approx(-2.1433630127818777, abs=1e-10)
