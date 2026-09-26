"""Scan a fixed multi-frame LF NOCI recipe across the Fig. 2b alpha grid."""

import csv
from pathlib import Path

import numpy as np

from scripts.fig2b_hf_scan import DEFAULT_ALPHA_VALUES
from src import lf_mp, lf_mr, my_direct_ep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "fig2b_mr.csv"
FRAME_RECIPE = "best_full_lf_orbit_plus_cs_start_orbit"
CSV_COLUMNS = (
    "alpha", "g", "omega", "L", "t",
    "nrandom", "random_scale", "seed", "gtol", "max_cycle",
    "coupling_convention", "frame_recipe",
    "lf_energy", "cs_start_energy", "lf_nconverged",
    "mr_orbit_energy", "mr_energy", "mr_orbit_rank", "mr_rank",
    "mr_n0", "mr_n1", "mr_n2", "mr_n3",
    "mr_density_imbalance", "mr_norm_error",
)


def _csv_row(record, *, nrandom, random_scale, seed, gtol, max_cycle):
    density = np.asarray(record["mr_density"])
    return {
        "alpha": f"{record['alpha']:.10g}",
        "g": f"{record['g']:.16g}",
        "omega": "0.5",
        "L": 4,
        "t": "-1",
        "nrandom": nrandom,
        "random_scale": f"{random_scale:.16g}",
        "seed": seed,
        "gtol": f"{gtol:.16g}",
        "max_cycle": max_cycle,
        "coupling_convention": "paper_uncentered",
        "frame_recipe": FRAME_RECIPE,
        "lf_energy": f"{record['lf_energy']:.16g}",
        "cs_start_energy": f"{record['cs_start_energy']:.16g}",
        "lf_nconverged": record["lf_nconverged"],
        "mr_orbit_energy": f"{record['mr_orbit_energy']:.16g}",
        "mr_energy": f"{record['mr_energy']:.16g}",
        "mr_orbit_rank": record["mr_orbit_rank"],
        "mr_rank": record["mr_rank"],
        "mr_n0": f"{density[0]:.16g}",
        "mr_n1": f"{density[1]:.16g}",
        "mr_n2": f"{density[2]:.16g}",
        "mr_n3": f"{density[3]:.16g}",
        "mr_density_imbalance": f"{record['mr_density_imbalance']:.16g}",
        "mr_norm_error": f"{record['mr_norm_error']:.16g}",
    }


def run_mr_scan(
    alpha_values: np.ndarray,
    *,
    nrandom: int = 8,
    random_scale: float = 0.5,
    seed: int = 0,
    gtol: float = 1e-8,
    max_cycle: int = 500,
    overlap_cut: float = 1e-10,
    csv_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Evaluate the fixed best-orbit plus CS-start-orbit NOCI baseline.

    Uses the same four-site LF-HF multistart settings as the existing
    Fig. 2b scan.  Each alpha is optimized independently.  This fixed
    recipe is a reproducible baseline, not an automatic reference search.
    Completed CSV rows are flushed after each alpha.
    """
    if alpha_values.ndim != 1 or alpha_values.size == 0:
        raise ValueError("alpha_values must be a nonempty one-dimensional array")
    if (np.iscomplexobj(alpha_values) or not np.all(np.isfinite(alpha_values))
            or np.any(alpha_values < 0)):
        raise ValueError("alpha_values must be real, finite, and nonnegative")

    length = 4
    omega = 0.5
    tmat = my_direct_ep.electron_ring_hopping(length, -1.0)
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
    try:
        for alpha_value in alpha_values:
            alpha = float(alpha_value)
            g = float(my_direct_ep.alpha_to_g(alpha, omega))
            best, results = lf_mp.lf_hf_full_alpha_point(
                alpha, tmat, omega, nrandom=nrandom,
                random_scale=random_scale, seed=seed, gtol=gtol,
                max_cycle=max_cycle,
            )
            cs_start = results[0]
            if not cs_start.success:
                raise RuntimeError(f"CS-start LF-HF branch failed at alpha={alpha}")

            base_lam, base_shift = lf_mr.lf_translation_orbit(
                best.lam, best.shift
            )
            orbit_energy, mr_energy, orbit_rank, mr_rank = (
                lf_mr.lf_noci_trial_orbit(
                    tmat, g, omega, base_lam, base_shift,
                    cs_start.lam, cs_start.shift,
                    overlap_cut=overlap_cut,
                )
            )
            added_lam, added_shift = lf_mr.lf_translation_orbit(
                cs_start.lam, cs_start.shift
            )
            all_lam = np.concatenate((base_lam, added_lam), axis=0)
            all_shift = np.concatenate((base_shift, added_shift), axis=0)
            smat = lf_mr.lf_frame_overlap(all_lam, all_shift)
            hmat = lf_mr.lf_frame_hamiltonian(
                tmat, g, omega, all_lam, all_shift
            )
            energy_check, coeff, rank_check = lf_mr.lf_noci_lowest(
                hmat, smat, overlap_cut=overlap_cut
            )
            if abs(energy_check - mr_energy) > 1e-10 or rank_check != mr_rank:
                raise RuntimeError(f"inconsistent NOCI solve at alpha={alpha}")
            density = lf_mr.lf_noci_site_density(coeff, smat)
            norm_error = abs(float(np.sum(density)) - 1.0)
            if orbit_energy > float(best.fun) + 1e-8:
                raise RuntimeError(f"orbit energy exceeds LF-HF at alpha={alpha}")
            if mr_energy > orbit_energy + 1e-8:
                raise RuntimeError(f"augmented energy exceeds orbit at alpha={alpha}")
            if norm_error > 1e-9:
                raise RuntimeError(f"NOCI state is not normalized at alpha={alpha}")

            record = {
                "alpha": alpha,
                "g": g,
                "lf_energy": float(best.fun),
                "cs_start_energy": float(cs_start.fun),
                "lf_nconverged": int(best.nconverged),
                "mr_orbit_energy": float(orbit_energy),
                "mr_energy": float(mr_energy),
                "mr_orbit_rank": int(orbit_rank),
                "mr_rank": int(mr_rank),
                "mr_density": density.copy(),
                "mr_density_imbalance": float(np.ptp(density)),
                "mr_norm_error": norm_error,
            }
            records.append(record)
            print(
                f"alpha={alpha:.1f} LF-HF={best.fun:.9f} "
                f"MR-LF={mr_energy:.9f} rank={mr_rank}",
                flush=True,
            )
            if writer is not None and csv_file is not None:
                writer.writerow(
                    _csv_row(
                        record, nrandom=nrandom, random_scale=random_scale,
                        seed=seed, gtol=gtol, max_cycle=max_cycle,
                    )
                )
                csv_file.flush()
    finally:
        if csv_file is not None:
            csv_file.close()
    return records


if __name__ == "__main__":
    completed = run_mr_scan(DEFAULT_ALPHA_VALUES, csv_path=DEFAULT_CSV_PATH)
    print(f"Saved {len(completed)} points to {DEFAULT_CSV_PATH}")
