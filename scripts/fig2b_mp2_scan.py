"""Scan CS-HF/MP2 and full-matrix LF-HF/MP2 branches for Fig. 2b."""

import csv
from pathlib import Path

import numpy as np

from scripts.fig2b_hf_scan import CSV_COLUMNS as HF_CSV_COLUMNS
from scripts.fig2b_hf_scan import DEFAULT_ALPHA_VALUES
from scripts.fig2b_hf_scan import _csv_row as _hf_csv_row
from src import cs_mp, lf_mp, my_direct_ep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "fig2b_mp2.csv"
DEFAULT_MAX_TOTAL = 10
CSV_COLUMNS = (
    *HF_CSV_COLUMNS,
    "lf_mp2_max_total",
    "lf_mp2_nconfig",
    "lf_mp2_pure_correction",
    "lf_mp2_single_correction",
    "lf_mp2_correction",
    "lf_mp2_energy",
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
    """Convert one MP2 scan record to a stable CSV row."""
    row = _hf_csv_row(
        record,
        omega=omega,
        norb=norb,
        hopping=hopping,
        nrandom=nrandom,
        random_scale=random_scale,
        seed=seed,
        gtol=gtol,
        max_cycle=max_cycle,
    )
    row.update(
        {
            "lf_mp2_max_total": record["lf_mp2_max_total"],
            "lf_mp2_nconfig": record["lf_mp2_nconfig"],
            "lf_mp2_pure_correction": (
                f"{record['lf_mp2_pure_correction']:.16g}"
            ),
            "lf_mp2_single_correction": (
                f"{record['lf_mp2_single_correction']:.16g}"
            ),
            "lf_mp2_correction": f"{record['lf_mp2_correction']:.16g}",
            "lf_mp2_energy": f"{record['lf_mp2_energy']:.16g}",
        }
    )
    return row


def run_mp2_scan(
    alpha_values: np.ndarray,
    *,
    max_total: int = DEFAULT_MAX_TOTAL,
    nrandom: int = 8,
    random_scale: float = 0.5,
    seed: int = 0,
    gtol: float = 1e-8,
    max_cycle: int = 500,
    csv_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Evaluate the Fig. 2b CS-HF/MP2 and full-matrix LF-HF/MP2 curves.

    The physical parameters are fixed to the four-site one-electron ring with
    ``t=-1`` and ``omega=0.5``.  LF-MP2 uses a collective total-phonon cutoff;
    the default value 10 is converged against 16 to better than ``1e-11`` on
    representative points spanning ``alpha=0.4`` through ``3.0``.

    Parameters
    ----------
    alpha_values
        Nonempty one-dimensional array of finite nonnegative couplings.  Input
        order and repeated values are preserved.
    max_total
        Positive inclusive LF-MP2 total-phonon cutoff.
    nrandom, random_scale, seed, gtol, max_cycle
        Full-matrix LF-HF multistart and optimization controls.
    csv_path
        Optional output path.  Completed rows are flushed immediately.

    Returns
    -------
    list of dict
        In-memory records containing the existing HF and CS-MP2 quantities
        plus LF-MP2 channel corrections, total correction, and total energy.
    """
    if not isinstance(max_total, (int, np.integer)) or isinstance(
        max_total, (bool, np.bool_)
    ):
        raise TypeError("max_total must be a positive integer")
    if max_total <= 0:
        raise ValueError("max_total must be a positive integer")
    if alpha_values.ndim != 1 or alpha_values.size == 0:
        raise ValueError("alpha_values must be a non-empty 1D array")
    if (
        np.iscomplexobj(alpha_values)
        or not np.all(np.isfinite(alpha_values))
        or np.any(alpha_values < 0)
    ):
        raise ValueError("alpha_values must be a real, finite, non-negative array")

    norb = 4
    omega = 0.5
    hopping = -1.0
    tmat = my_direct_ep.electron_ring_hopping(norb, hopping)
    cs_coeff0 = np.full(norb, 1.0 / np.sqrt(norb))
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
            try:
                cs_result = cs_mp.cs_mp2_point(
                    tmat,
                    g,
                    omega,
                    cs_coeff0,
                    conv_tol=gtol,
                    max_cycle=max_cycle,
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"CS-HF failed to converge for alpha={alpha}, g={g}"
                ) from exc
            best, _ = lf_mp.lf_mp2_alpha_point(
                alpha,
                tmat,
                omega,
                max_total=max_total,
                nrandom=nrandom,
                random_scale=random_scale,
                seed=seed,
                gtol=gtol,
                max_cycle=max_cycle,
            )
            cs_density = cs_mp.cs_site_density(np.asarray(cs_result["coeff"]))
            lf_density = np.asarray(best["density"])
            pure_correction = float(
                np.sum(
                    np.abs(best["pure_matrix_elements"]) ** 2
                    / best["pure_denominators"]
                )
            )
            single_correction = float(
                np.sum(
                    np.abs(best["single_matrix_elements"]) ** 2
                    / best["single_denominators"]
                )
            )
            record = {
                "alpha": alpha,
                "g": g,
                "cs_energy": float(cs_result["hf_energy"]),
                "cs_mp2_correction": float(cs_result["mp2_correction"]),
                "cs_mp2_energy": float(cs_result["total_energy"]),
                "cs_density": cs_density.copy(),
                "cs_density_imbalance": float(
                    np.max(cs_density) - np.min(cs_density)
                ),
                "cs_niter": int(cs_result["niter"]),
                "cs_density_residual": float(cs_result["density_residual"]),
                "lf_energy": float(best.fun),
                "lf_density": lf_density.copy(),
                "lf_density_imbalance": float(best["density_imbalance"]),
                "lf_nstart": int(best["nstart"]),
                "lf_nconverged": int(best["nconverged"]),
                "lf_shift_residual": float(best["shift_residual"]),
                "lf_mp2_max_total": int(max_total),
                "lf_mp2_nconfig": int(best["occupations"].shape[0]),
                "lf_mp2_pure_correction": pure_correction,
                "lf_mp2_single_correction": single_correction,
                "lf_mp2_correction": float(best["mp2_correction"]),
                "lf_mp2_energy": float(best["total_energy"]),
            }
            records.append(record)
            if writer is not None and csv_file is not None:
                writer.writerow(
                    _csv_row(
                        record,
                        omega=omega,
                        norb=norb,
                        hopping=hopping,
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
    completed = run_mp2_scan(DEFAULT_ALPHA_VALUES, csv_path=DEFAULT_CSV_PATH)
    print(f"Saved {len(completed)} points to {DEFAULT_CSV_PATH}")
