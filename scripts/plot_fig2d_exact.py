"""Plot converged four-electron ED curves in the parameter range of Fig. 2d."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.fig2d_exact_scan import COLUMNS, ROOT


def collect_results(paths):
    """Merge cutoff records and require a converged point for the full grid."""
    records = []
    for path in paths:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            records.extend(csv.DictReader(handle))
    groups = {}
    for row in records:
        row.setdefault("solver", "davidson")
        if (int(row["L"]) != 4 or int(row["neleca"]) != 2
                or int(row["nelecb"]) != 2 or float(row["U"]) != 4
                or row["convention"] != "centered_solve_paper_energy"):
            raise ValueError("incompatible Fig. 2d model")
        key = (float(row["omega"]), round(float(row["alpha"]), 10))
        groups.setdefault(key, []).append(row)
    expected = {(w, round(float(a), 10)) for w in (0.5, 5.0)
                for a in np.linspace(0, 4, 11)}
    if set(groups) != expected:
        raise ValueError(f"incomplete alpha grid: missing {sorted(expected-set(groups))}")
    selected = []
    for key in sorted(groups):
        sequence = sorted(groups[key], key=lambda r: int(r["Nmax"]))
        if len({int(r["Nmax"]) for r in sequence}) != len(sequence):
            raise ValueError(f"duplicated cutoffs at {key}")
        last = sequence[-1]
        if (last["cutoff_converged"] != "True"
                or last["solver_converged"] != "True"
                or float(last["residual"]) >= float(last["residual_tolerance"])):
            raise ValueError(f"unconverged point {key}")
        if not np.isclose(float(last["energy"]),
                          float(last["energy_centered"])-4*key[1],
                          atol=1e-10, rtol=0):
            raise ValueError("incorrect centering energy shift")
        selected.append(last)
    records.sort(key=lambda r: (float(r["omega"]), float(r["alpha"]), int(r["Nmax"])))
    return records, selected


def plot_scan(paths, output_dir=ROOT / "figures", data_dir=ROOT / "data"):
    """Save all cutoff data, final-point CSV, and energy/convergence plots."""
    records, selected = collect_results(paths)
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("fig2d_exact_convergence.csv", records),
                       ("fig2d_exact.csv", selected)):
        with (data_dir / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(5.5, 4.7), constrained_layout=True)
    for omega, color, marker in ((0.5, "tab:purple", "o"), (5., "tab:cyan", "s")):
        rows = [r for r in selected if float(r["omega"]) == omega]
        ax.plot([float(r["alpha"]) for r in rows],
                [float(r["energy"]) for r in rows], color=color,
                marker=marker, markerfacecolor="none", markersize=5,
                linewidth=1.3, label=rf"ED, $\omega={omega:g}$")
    ax.set(xlabel=r"$\alpha=g^2/\omega$", ylabel=r"$E/|t|$",
           xlim=(0, 4), ylim=(-26, -1),
           title=r"$L=4,\ N_e=4,\ U=4,\ t=-1$")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fig2d_exact.{suffix}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for ax, omega in zip(axes, (0.5, 5.0)):
        for alpha in (.8, 2., 3.2, 4.):
            rows = [r for r in records if float(r["omega"]) == omega
                    and np.isclose(float(r["alpha"]), alpha) and r["delta_E"]]
            ax.semilogy([int(r["Nmax"]) for r in rows],
                        [max(float(r["delta_E"]), 1e-14) for r in rows],
                        marker="o", markersize=3.5, linewidth=1.1,
                        label=rf"$\alpha={alpha:g}$")
        ax.axhline(1e-6, color="0.4", linestyle="--", linewidth=1,
                   label=r"$10^{-6}$ target")
        ax.set(xlabel=r"$N_{\max}$ per local mode",
               ylabel=r"$|E(N_{\max})-E(N_{\max}^{\rm previous})|$",
               title=rf"$\omega={omega:g}$")
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"fig2d_exact_cutoff_convergence.{suffix}", dpi=300)
    plt.close(fig)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()
    rows = plot_scan(args.inputs)
    print(f"Saved {len(rows)} converged points and both figures.")
