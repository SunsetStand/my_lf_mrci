"""Scan the CS-HF and local LF-HF branches needed for Fig. 2b."""

import csv
from pathlib import Path

import numpy as np

from src import cs_mp, lf_mp, my_direct_ep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "fig2b_hf.csv"
DEFAULT_ALPHA_VALUES = np.linspace(0.0, 3.0, 16)
CSV_COLUMNS = (
    "alpha",
    "g",
    "omega",
    "L",
    "t",
    "nrandom",
    "random_scale",
    "seed",
    "gtol",
    "max_cycle",
    "coupling_convention",
    "cs_branch",
    "lf_ansatz",
    "cs_energy",
    "cs_n0",
    "cs_n1",
    "cs_n2",
    "cs_n3",
    "cs_density_imbalance",
    "cs_niter",
    "cs_density_residual",
    "lf_energy",
    "lf_n0",
    "lf_n1",
    "lf_n2",
    "lf_n3",
    "lf_density_imbalance",
    "lf_nstart",
    "lf_nconverged",
    "lf_shift_residual",
)


def _csv_row(
    record: dict[str, object],
    *,
    omega: float,
    norb: int,
    hopping: float,
    nrandom: int,
    random_scale: float,
    seed: int,
    gtol: float,
    max_cycle: int,
) -> dict[str, object]:
    """Convert one in-memory record to a stable CSV row."""
    cs_density = np.asarray(record["cs_density"])
    lf_density = np.asarray(record["lf_density"])
    return {
        "alpha": f"{record['alpha']:.10g}",
        "g": f"{record['g']:.16g}",
        "omega": f"{omega:.16g}",
        "L": norb,
        "t": f"{hopping:.16g}",
        "nrandom": nrandom,
        "random_scale": f"{random_scale:.16g}",
        "seed": seed,
        "gtol": f"{gtol:.16g}",
        "max_cycle": max_cycle,
        "coupling_convention": "paper_uncentered",
        "cs_branch": "uniform_symmetric",
        "lf_ansatz": "local_diagonal",
        "cs_energy": f"{record['cs_energy']:.16g}",
        "cs_n0": f"{cs_density[0]:.16g}",
        "cs_n1": f"{cs_density[1]:.16g}",
        "cs_n2": f"{cs_density[2]:.16g}",
        "cs_n3": f"{cs_density[3]:.16g}",
        "cs_density_imbalance": f"{record['cs_density_imbalance']:.16g}",
        "cs_niter": record["cs_niter"],
        "cs_density_residual": f"{record['cs_density_residual']:.16g}",
        "lf_energy": f"{record['lf_energy']:.16g}",
        "lf_n0": f"{lf_density[0]:.16g}",
        "lf_n1": f"{lf_density[1]:.16g}",
        "lf_n2": f"{lf_density[2]:.16g}",
        "lf_n3": f"{lf_density[3]:.16g}",
        "lf_density_imbalance": f"{record['lf_density_imbalance']:.16g}",
        "lf_nstart": record["lf_nstart"],
        "lf_nconverged": record["lf_nconverged"],
        "lf_shift_residual": f"{record['lf_shift_residual']:.16g}",
    }


def run_hf_scan(
    alpha_values: np.ndarray,
    *,
    nrandom: int = 8,
    random_scale: float = 0.5,
    seed: int = 0,
    gtol: float = 1e-8,
    max_cycle: int = 500,
    csv_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Evaluate CS-HF and local LF-HF in the supplied coupling order.

    This experiment-level driver fixes the Fig. 2b model parameters to a
    four-site ring with ``t=-1`` and ``omega=0.5``.  CS-HF is initialized on
    its uniform branch, while every LF-HF point uses the same reproducible
    set of CS-centered random perturbations.

    Parameters
    ----------
    alpha_values
        Nonempty one-dimensional array of finite nonnegative couplings.
        Values are neither sorted nor deduplicated.
    nrandom
        Number of random LF-HF starts in addition to the CS point.
    random_scale
        Standard deviation of the LF-HF starting-point perturbations.
    seed
        Seed reused at every coupling point.
    gtol
        Convergence tolerance passed to the CS-HF and LF-HF drivers.
    max_cycle
        Maximum iteration count passed to both drivers.
    csv_path
        Optional output path.  The header is written before the scan and
        each completed point is flushed immediately.

    Returns
    -------
    list of dict
        One record per input coupling, in input order, containing the CS-HF
        and lowest successful LF-HF energies, densities, residuals, and
        convergence metadata.

    Raises
    ------
    RuntimeError
        If the symmetric CS-HF calculation fails to converge at any point.
    """
    norb = 4
    omega = 0.5
    t = -1.0
    tmat = my_direct_ep.electron_ring_hopping(norb, t)
    cs_coeff0 = np.full(norb, 1.0 / np.sqrt(norb))
    if alpha_values.ndim != 1 or alpha_values.size == 0:
        raise ValueError("alpha_values must be a non-empty 1D array")
    if (
        np.iscomplexobj(alpha_values)
        or not np.all(np.isfinite(alpha_values))
        or np.any(alpha_values < 0)
    ):
        raise ValueError("alpha_values must be a real, finite, non-negative array")
    csv_file = None
    writer = None
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        csv_file.flush()

    records = []
    try:
        for alpha in alpha_values:
            alpha = float(alpha)
            g = float(my_direct_ep.alpha_to_g(alpha, omega))
            (
                cs_energy,
                cs_coeff,
                _,
                cs_converged,
                cs_niter,
                cs_density_residual,
            ) = cs_mp.cs_hf_scf(
                tmat,
                g,
                omega,
                cs_coeff0,
                conv_tol=gtol,
                max_cycle=max_cycle,
            )
            if not cs_converged:
                raise RuntimeError(f"CS-HF failed to converge for alpha={alpha}, g={g}")
            cs_density = cs_mp.cs_site_density(cs_coeff)
            cs_density_imbalance = float(np.max(cs_density) - np.min(cs_density))
            best, _ = lf_mp.lf_hf_local_alpha_point(
                alpha,
                tmat,
                omega,
                nrandom=nrandom,
                random_scale=random_scale,
                seed=seed,
                gtol=gtol,
                max_cycle=max_cycle,
            )
            record = {
                "alpha": alpha,
                "g": g,
                "cs_energy": float(cs_energy),
                "cs_density": cs_density.copy(),
                "cs_density_imbalance": cs_density_imbalance,
                "cs_niter": cs_niter,
                "cs_density_residual": cs_density_residual,
                "lf_energy": float(best.fun),
                "lf_density": best["density"].copy(),
                "lf_density_imbalance": best["density_imbalance"],
                "lf_nstart": best["nstart"],
                "lf_nconverged": best["nconverged"],
                "lf_shift_residual": best["shift_residual"],
            }
            records.append(record)

            if writer is not None and csv_file is not None:
                writer.writerow(
                    _csv_row(
                        record,
                        omega=omega,
                        norb=norb,
                        hopping=t,
                        nrandom=nrandom,
                        random_scale=random_scale,
                        seed=seed,
                        gtol=gtol,
                        max_cycle=max_cycle,
                    )
                )
                csv_file.flush()
    finally:
        if csv_file is not None:
            csv_file.close()

    return records


if __name__ == "__main__":
    completed = run_hf_scan(DEFAULT_ALPHA_VALUES, csv_path=DEFAULT_CSV_PATH)
    print(f"Saved {len(completed)} points to {DEFAULT_CSV_PATH}")
