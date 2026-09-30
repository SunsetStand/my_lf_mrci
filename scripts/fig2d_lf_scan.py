"""Scan Fig. 2d unrestricted LF-HF and LF-MP2 and compare with existing ED."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.lf_mp import lf_hf_multi_optimize, lf_mp2_unrestricted_reference_point


ROOT = Path(__file__).resolve().parents[1]
NELEC = (2, 2)
NSITE = 4
U = 4.0
OMEGAS = (0.5, 5.0)
ALPHAS = tuple(np.linspace(0.0, 4.0, 11))
BRANCH_COLUMNS = (
    "omega", "alpha", "start", "hf_energy", "success",
    "optimizer_success", "residual_ok", "orbital_grad_norm",
    "lam_grad_max", "total_density", "spin_density",
    "orbital_gap_alpha", "orbital_gap_beta", "nit", "message",
)
POINT_COLUMNS = (
    "omega", "alpha", "g", "hf_energy", "mp2_correction", "lf_mp2_energy",
    "ed_energy", "hf_minus_ed", "mp2_minus_ed", "selected_start",
    "orbital_grad_norm", "lam_grad_max", "orbital_gap_alpha",
    "orbital_gap_beta", "total_density", "spin_density", "mp2_max_total",
    "mp2_configurations", "minimum_denominator", "mp2_nph11",
    "mp2_nph13",
)


def hopping():
    eye = np.eye(NSITE)
    return -(np.roll(eye, 1, axis=1) + np.roll(eye, -1, axis=1))


def initial_references(g, omega, previous=None):
    """Yield AFM, charge ordered, and preceding-point full-matrix starts."""
    eye = np.eye(NSITE)
    afm_alpha = eye[:, [0, 2, 1, 3]]
    afm_beta = eye[:, [1, 3, 0, 2]]
    cdw = afm_alpha
    displacement = eye * (g / omega)
    yield "afm", afm_alpha, afm_beta, displacement
    yield "cdw", cdw, cdw, displacement
    if previous is not None:
        yield "previous", previous.mo_coeff[0], previous.mo_coeff[1], previous.lam


def orbital_gaps(result):
    energies = result.mo_energy
    return tuple(
        float(np.min(energies[s, NELEC[s]:]) -
              np.max(energies[s, :NELEC[s]]))
        for s in range(2)
    )


def density_string(array):
    return " ".join(f"{value:.12g}" for value in np.asarray(array).ravel())


def read_exact(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(omega, round(float(alpha), 10))
                for omega in OMEGAS for alpha in ALPHAS}
    selected = {}
    for row in rows:
        key = (float(row["omega"]), round(float(row["alpha"]), 10))
        if (int(row["L"]) != NSITE
                or (int(row["neleca"]), int(row["nelecb"])) != NELEC
                or float(row["U"]) != U
                or row["cutoff_converged"] != "True"
                or row["solver_converged"] != "True"):
            raise ValueError(f"incompatible or unconverged ED row: {key}")
        if key in selected:
            raise ValueError(f"duplicate ED row: {key}")
        selected[key] = float(row["energy"])
    if set(selected) != expected:
        raise ValueError(f"incomplete ED grid: {sorted(expected - set(selected))}")
    return selected


def write_csv(path, columns, records):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def run_scan(exact_path=ROOT / "data/fig2d_exact.csv",
             output_dir=ROOT / "data", figure_dir=ROOT / "figures"):
    """Run and persist every branch, selected reference, correction, and plot."""
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    exact = read_exact(exact_path)
    tmat = hopping()
    branch_rows = []
    point_rows = []
    references = []
    for omega in OMEGAS:
        previous = None
        for alpha in ALPHAS:
            g = float(np.sqrt(alpha * omega))
            candidates = []
            for name, coeff_a, coeff_b, lam0 in initial_references(
                g, omega, previous,
            ):
                try:
                    result = lf_hf_multi_optimize(
                        tmat, U, g, omega, NELEC,
                        coeff_a, coeff_b, lam0,
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                    branch_rows.append({
                        "omega": omega, "alpha": alpha, "start": name,
                        "hf_energy": "", "success": False,
                        "optimizer_success": False, "residual_ok": False,
                        "orbital_grad_norm": "", "lam_grad_max": "",
                        "total_density": "", "spin_density": "",
                        "orbital_gap_alpha": "", "orbital_gap_beta": "",
                        "nit": "", "message": str(exc),
                    })
                    print(f"FAILED omega={omega:g} alpha={alpha:g} "
                          f"start={name}: {exc}", flush=True)
                    continue
                gaps = orbital_gaps(result)
                branch_rows.append({
                    "omega": omega, "alpha": alpha, "start": name,
                    "hf_energy": float(result.fun),
                    "success": bool(result.success),
                    "optimizer_success": bool(result.optimizer_success),
                    "residual_ok": bool(result.residual_ok),
                    "orbital_grad_norm": result.orbital_grad_norm,
                    "lam_grad_max": result.lam_grad_max,
                    "total_density": density_string(
                        result.spin_density.sum(axis=0)
                    ),
                    "spin_density": density_string(result.spin_density),
                    "orbital_gap_alpha": gaps[0],
                    "orbital_gap_beta": gaps[1],
                    "nit": result.nit, "message": str(result.message),
                })
                if result.success:
                    candidates.append((name, result))
            if not candidates:
                previous = None
                print(f"UNCONVERGED omega={omega:g} alpha={alpha:g}", flush=True)
                continue
            name, best = min(candidates, key=lambda pair: pair[1].fun)
            previous = best
            try:
                mp2 = lf_mp2_unrestricted_reference_point(
                    tmat, U, g, omega, NELEC,
                    best.mo_coeff, best.mo_energy, best.lam, best.shift,
                    max_total=8, max_excited_modes=2,
                )
            except ValueError as exc:
                print(f"MP2 FAILED omega={omega:g} alpha={alpha:g}: {exc}",
                      flush=True)
                continue
            cutoffs = {}
            if round(float(alpha), 10) in (0.0, 1.6, 3.2, 4.0):
                for nph, max_total in ((11, 10), (13, 12)):
                    cutoff_result = lf_mp2_unrestricted_reference_point(
                        tmat, U, g, omega, NELEC,
                        best.mo_coeff, best.mo_energy, best.lam, best.shift,
                        max_total=max_total, max_excited_modes=2,
                    )
                    cutoffs[f"mp2_nph{nph}"] = cutoff_result["total_energy"]
            ed_energy = exact[(omega, round(float(alpha), 10))]
            gaps = orbital_gaps(best)
            point_rows.append({
                "omega": omega, "alpha": alpha, "g": g,
                "hf_energy": float(best.fun),
                "mp2_correction": mp2["mp2_correction"],
                "lf_mp2_energy": mp2["total_energy"],
                "ed_energy": ed_energy,
                "hf_minus_ed": float(best.fun) - ed_energy,
                "mp2_minus_ed": mp2["total_energy"] - ed_energy,
                "selected_start": name,
                "orbital_grad_norm": best.orbital_grad_norm,
                "lam_grad_max": best.lam_grad_max,
                "orbital_gap_alpha": gaps[0],
                "orbital_gap_beta": gaps[1],
                "total_density": density_string(best.spin_density.sum(axis=0)),
                "spin_density": density_string(best.spin_density),
                "mp2_max_total": 8,
                "mp2_configurations": len(mp2["occupations"]),
                "minimum_denominator": mp2["minimum_denominator"],
                "mp2_nph11": cutoffs.get("mp2_nph11", ""),
                "mp2_nph13": cutoffs.get("mp2_nph13", ""),
            })
            references.append(best)
            print(f"OK omega={omega:g} alpha={alpha:g} {name} "
                  f"HF={best.fun:.10f} MP2={mp2['total_energy']:.10f}",
                  flush=True)
    write_csv(output_dir / "fig2d_lf_branches.csv",
              BRANCH_COLUMNS, branch_rows)
    write_csv(output_dir / "fig2d_lf.csv", POINT_COLUMNS, point_rows)
    if len(point_rows) != len(OMEGAS) * len(ALPHAS):
        raise RuntimeError(
            f"only {len(point_rows)} of {len(OMEGAS)*len(ALPHAS)} points "
            "passed HF and MP2 checks; no complete figure was produced"
        )
    np.savez_compressed(
        output_dir / "fig2d_lf_references.npz",
        omega=np.array([row["omega"] for row in point_rows]),
        alpha=np.array([row["alpha"] for row in point_rows]),
        mo_coeff=np.stack([ref.mo_coeff for ref in references]),
        mo_energy=np.stack([ref.mo_energy for ref in references]),
        lam=np.stack([ref.lam for ref in references]),
        spin_density=np.stack([ref.spin_density for ref in references]),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2),
                             constrained_layout=True, sharey=True)
    for ax, omega in zip(axes, OMEGAS):
        rows = [row for row in point_rows if row["omega"] == omega]
        x = [row["alpha"] for row in rows]
        for field, label, marker in (
            ("ed_energy", "ED", "o"),
            ("hf_energy", "LF-HF", "s"),
            ("lf_mp2_energy", "LF-MP2", "^"),
        ):
            ax.plot(x, [row[field] for row in rows], marker=marker,
                    markersize=3.5, linewidth=1.3, label=label)
        ax.set(xlabel="alpha = g² / omega", title=f"omega = {omega:g}",
               xlim=(0, 4))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylabel("Energy / |t|")
    fig.suptitle("Four-site Hubbard-Holstein, four electrons, U = 4")
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"fig2d_lf.{suffix}", dpi=300)
    plt.close(fig)
    return point_rows, branch_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path,
                        default=ROOT / "data/fig2d_exact.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    run_scan(args.exact, args.output_dir, args.figure_dir)
