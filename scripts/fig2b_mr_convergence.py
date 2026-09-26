"""Track MR-LF energy and overlap rank as frame orbits are added."""

import csv
from pathlib import Path

import numpy as np

from scripts.fig2b_hf_scan import DEFAULT_CSV_PATH as DEFAULT_HF_CSV_PATH
from scripts.plot_fig2b_exact import DEFAULT_CSV_PATH as DEFAULT_EXACT_CSV_PATH
from scripts.plot_fig2b_exact import read_exact_csv
from scripts.plot_fig2b_hf import read_hf_csv
from src import lf_mp, lf_mr, my_direct_ep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "fig2b_mr_worst_convergence.csv"
DEFAULT_THETA_POOL = (0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0)
CSV_COLUMNS = (
    "alpha", "g", "omega", "L", "t", "nrandom", "random_scale",
    "seed", "gtol", "max_cycle", "overlap_cut",
    "overlap_cut_loose", "overlap_cut_tight", "min_gain",
    "theta_pool", "lf_energy_reference", "exact_energy",
    "step", "phase", "selected_theta", "n_frames", "rank",
    "energy", "gap_exact", "density_imbalance", "min_retained_overlap",
    "energy_cut_loose", "rank_cut_loose",
    "energy_cut_tight", "rank_cut_tight",
)


def _measure(tmat, g, omega, lam, shift, overlap_cut):
    smat = lf_mr.lf_frame_overlap(lam, shift)
    hmat = lf_mr.lf_frame_hamiltonian(tmat, g, omega, lam, shift)
    energy, coeff, rank = lf_mr.lf_noci_lowest(
        hmat, smat, overlap_cut=overlap_cut
    )
    density = lf_mr.lf_noci_site_density(coeff, smat)
    if abs(float(np.sum(density)) - 1.0) > 1e-9:
        raise RuntimeError("NOCI state is not normalized")
    eigenvalues = np.linalg.eigvalsh(smat.reshape(coeff.size, coeff.size))
    threshold = overlap_cut * max(1.0, float(eigenvalues[-1]))
    retained = eigenvalues[eigenvalues > threshold]
    if retained.size != rank:
        raise RuntimeError("retained overlap rank is inconsistent")
    return (
        float(energy), int(rank), float(np.ptp(density)),
        float(retained[0]),
    )


def run_worst_convergence(
    *,
    hf_csv_path: str | Path = DEFAULT_HF_CSV_PATH,
    exact_csv_path: str | Path = DEFAULT_EXACT_CSV_PATH,
    theta_pool: tuple[float, ...] = DEFAULT_THETA_POOL,
    nrandom: int = 8,
    random_scale: float = 0.5,
    seed: int = 0,
    gtol: float = 1e-8,
    max_cycle: int = 500,
    overlap_cut: float = 1e-10,
    min_gain: float = 1e-8,
    csv_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Greedily add complete frame orbits at the worst LF-HF error point.

    The candidate pool is a fixed linear path in the full LF parameter
    matrices between the CS-start and best symmetry-broken LF-HF frames.
    This finite-pool diagnostic does not optimize frame parameters or
    establish convergence to the exact Hilbert space.
    """
    hf, hf_meta = read_hf_csv(hf_csv_path)
    exact_alpha, exact_energy, exact_meta = read_exact_csv(exact_csv_path)
    if (
        int(hf_meta["L"]) != 4 or int(exact_meta["L"]) != 4
        or not np.isclose(float(hf_meta["omega"]), 0.5)
        or not np.isclose(float(exact_meta["omega"]), 0.5)
        or not np.isclose(float(hf_meta["t"]), -1.0)
        or hf_meta["coupling_convention"] != "paper_uncentered"
    ):
        raise ValueError("reference CSV files have incompatible model conventions")
    if (len(hf["alpha"]) != len(exact_alpha)
            or not np.allclose(hf["alpha"], exact_alpha, atol=1e-12, rtol=0)):
        raise ValueError("HF and exact alpha grids must match")
    if (not np.isfinite(overlap_cut) or overlap_cut <= 0
            or not np.isfinite(min_gain) or min_gain < 0):
        raise ValueError("overlap_cut and min_gain must be finite and valid")
    thetas = np.asarray(theta_pool, dtype=np.float64)
    if thetas.ndim != 1 or thetas.size == 0 or not np.all(np.isfinite(thetas)):
        raise ValueError("theta_pool must be a nonempty finite sequence")
    if len(np.unique(thetas)) != len(thetas):
        raise ValueError("theta_pool must not contain duplicate parameters")

    worst = int(np.argmax(hf["lf_energy"] - exact_energy))
    alpha = float(hf["alpha"][worst])
    exact = float(exact_energy[worst])
    reference_lf = float(hf["lf_energy"][worst])
    omega = 0.5
    g = float(my_direct_ep.alpha_to_g(alpha, omega))
    tmat = my_direct_ep.electron_ring_hopping(4, -1.0)
    best, results = lf_mp.lf_hf_full_alpha_point(
        alpha, tmat, omega, nrandom=nrandom,
        random_scale=random_scale, seed=seed, gtol=gtol,
        max_cycle=max_cycle,
    )
    if abs(float(best.fun) - reference_lf) > 1e-8:
        raise RuntimeError("reoptimized LF-HF energy differs from CSV reference")
    cs_start = results[0]
    if not cs_start.success:
        raise RuntimeError("CS-start LF-HF branch did not converge")
    seeds = np.stack([
        (1.0 - theta) * cs_start.lam + theta * best.lam
        for theta in thetas
    ])
    seed_shifts = np.stack([
        (1.0 - theta) * cs_start.shift + theta * best.shift
        for theta in thetas
    ])
    theta_grid = ",".join(f"{theta:g}" for theta in thetas)

    csv_file = None
    writer = None
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        csv_file.flush()

    records = []
    def add_record(phase, chosen_theta, lam, shift):
        energy, rank, imbalance, smallest = _measure(
            tmat, g, omega, lam, shift, overlap_cut
        )
        loose_cut = overlap_cut * 100.0
        tight_cut = overlap_cut / 100.0
        energy_loose, rank_loose, _, _ = _measure(
            tmat, g, omega, lam, shift, loose_cut
        )
        energy_tight, rank_tight, _, _ = _measure(
            tmat, g, omega, lam, shift, tight_cut
        )
        if energy < exact - 1e-6:
            raise RuntimeError("NOCI energy falls below converged exact reference")
        if records and energy > records[-1]["energy"] + 1e-8:
            raise RuntimeError("adding frames unexpectedly raised NOCI energy")
        record = {
            "alpha": alpha, "g": g, "omega": omega, "L": 4, "t": -1.0,
            "nrandom": nrandom, "random_scale": random_scale,
            "seed": seed, "gtol": gtol, "max_cycle": max_cycle,
            "overlap_cut": overlap_cut,
            "overlap_cut_loose": loose_cut,
            "overlap_cut_tight": tight_cut, "min_gain": min_gain,
            "theta_pool": theta_grid, "lf_energy_reference": reference_lf,
            "exact_energy": exact, "step": len(records),
            "phase": phase, "selected_theta": chosen_theta,
            "n_frames": int(lam.shape[0]), "rank": rank,
            "energy": energy, "gap_exact": energy - exact,
            "density_imbalance": imbalance,
            "min_retained_overlap": smallest,
            "energy_cut_loose": energy_loose, "rank_cut_loose": rank_loose,
            "energy_cut_tight": energy_tight, "rank_cut_tight": rank_tight,
        }
        records.append(record)
        print(
            f"step={record['step']} phase={phase} K={lam.shape[0]} "
            f"rank={rank} E={energy:.12f} gap={energy-exact:.3e}",
            flush=True,
        )
        if writer is not None and csv_file is not None:
            row = {
                key: ("" if value is None else
                      f"{value:.16g}" if isinstance(value, float) else value)
                for key, value in record.items()
            }
            writer.writerow(row)
            csv_file.flush()
        return energy, rank

    try:
        single_lam = best.lam[None, :, :]
        single_shift = best.shift[None, :]
        add_record("best_lf_hf_frame", None, single_lam, single_shift)

        base_lam, base_shift = lf_mr.lf_translation_orbit(
            best.lam, best.shift
        )
        add_record("best_translation_orbit", None, base_lam, base_shift)

        sym_lam, sym_shift = lf_mr.lf_translation_orbit(
            cs_start.lam, cs_start.shift
        )
        base_lam = np.concatenate((base_lam, sym_lam), axis=0)
        base_shift = np.concatenate((base_shift, sym_shift), axis=0)
        add_record("plus_cs_start_orbit", None, base_lam, base_shift)

        while len(thetas):
            base_energy, base_rank, trial_energy, trial_rank = (
                lf_mr.lf_noci_score_orbits(
                    tmat, g, omega, base_lam, base_shift,
                    seeds, seed_shifts, overlap_cut=overlap_cut,
                )
            )
            if (abs(base_energy - records[-1]["energy"]) > 1e-9
                    or base_rank != records[-1]["rank"]):
                raise RuntimeError("candidate scores use an inconsistent base")
            eligible = (
                (trial_rank > base_rank)
                & (base_energy - trial_energy > min_gain)
            )
            if not np.any(eligible):
                break
            chosen = int(np.argmin(np.where(eligible, trial_energy, np.inf)))
            theta = float(thetas[chosen])
            orbit_lam, orbit_shift = lf_mr.lf_translation_orbit(
                seeds[chosen], seed_shifts[chosen]
            )
            base_lam = np.concatenate((base_lam, orbit_lam), axis=0)
            base_shift = np.concatenate((base_shift, orbit_shift), axis=0)
            measured_energy, measured_rank = add_record(
                "greedy_candidate_orbit", theta, base_lam, base_shift
            )
            if (abs(measured_energy - trial_energy[chosen]) > 1e-8
                    or measured_rank != trial_rank[chosen]):
                raise RuntimeError("selected trial score differs from combined solve")
            thetas = np.delete(thetas, chosen)
            seeds = np.delete(seeds, chosen, axis=0)
            seed_shifts = np.delete(seed_shifts, chosen, axis=0)
    finally:
        if csv_file is not None:
            csv_file.close()
    return records


if __name__ == "__main__":
    completed = run_worst_convergence(csv_path=DEFAULT_CSV_PATH)
    print(f"Saved {len(completed)} stages to {DEFAULT_CSV_PATH}")
