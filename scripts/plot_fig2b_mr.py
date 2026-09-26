"""Overlay fixed-recipe MR-LF NOCI on the existing Fig. 2b comparison."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.fig2b_mr_scan import DEFAULT_CSV_PATH as DEFAULT_MR_CSV_PATH
from scripts.plot_fig2b_exact import DEFAULT_CSV_PATH as DEFAULT_EXACT_CSV_PATH
from scripts.plot_fig2b_exact import read_exact_csv
from scripts.plot_fig2b_mp2 import DEFAULT_MP2_CSV_PATH, read_mp2_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PNG_PATH = PROJECT_ROOT / "figures" / "fig2b_mr.png"
DEFAULT_PDF_PATH = PROJECT_ROOT / "figures" / "fig2b_mr.pdf"


def read_mr_csv(csv_path: str | Path = DEFAULT_MR_CSV_PATH):
    """Read the separate fixed-recipe MR scan and sort it by alpha."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("MR scan CSV is empty")
    required = {
        "alpha", "omega", "L", "t", "coupling_convention", "frame_recipe",
        "mr_energy", "mr_orbit_energy", "mr_rank", "mr_density_imbalance",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"MR scan CSV is missing columns: {sorted(missing)}")
    metadata = {key: rows[0][key] for key in (
        "omega", "L", "t", "coupling_convention", "frame_recipe",
    )}
    for row in rows:
        if any(row[key] != value for key, value in metadata.items()):
            raise ValueError("MR scan CSV mixes model conventions or frame recipes")
    rows.sort(key=lambda row: float(row["alpha"]))
    data = {
        "alpha": np.array([float(row["alpha"]) for row in rows]),
        "mr_energy": np.array([float(row["mr_energy"]) for row in rows]),
        "mr_orbit_energy": np.array(
            [float(row["mr_orbit_energy"]) for row in rows]
        ),
        "mr_rank": np.array([int(row["mr_rank"]) for row in rows], dtype=np.int64),
        "mr_density_imbalance": np.array(
            [float(row["mr_density_imbalance"]) for row in rows]
        ),
    }
    if any(not np.all(np.isfinite(values)) for values in data.values()):
        raise ValueError("MR scan CSV contains nonfinite values")
    return data, metadata


def plot_mr(
    mr_csv_path: str | Path = DEFAULT_MR_CSV_PATH,
    mp2_csv_path: str | Path = DEFAULT_MP2_CSV_PATH,
    exact_csv_path: str | Path = DEFAULT_EXACT_CSV_PATH,
    png_path: str | Path = DEFAULT_PNG_PATH,
    pdf_path: str | Path = DEFAULT_PDF_PATH,
) -> tuple[Path, Path]:
    """Plot existing Fig. 2b curves plus the fixed-frame MR-LF baseline."""
    mp2, mp2_meta = read_mp2_csv(mp2_csv_path)
    mr, mr_meta = read_mr_csv(mr_csv_path)
    exact_alpha, exact_energy, exact_meta = read_exact_csv(exact_csv_path)
    if (
        int(mp2_meta["L"]) != int(mr_meta["L"])
        or int(exact_meta["L"]) != int(mr_meta["L"])
        or not np.isclose(float(mp2_meta["omega"]), float(mr_meta["omega"]))
        or not np.isclose(float(exact_meta["omega"]), float(mr_meta["omega"]))
        or not np.isclose(float(mp2_meta["t"]), float(mr_meta["t"]))
        or mp2_meta["coupling_convention"] != mr_meta["coupling_convention"]
    ):
        raise ValueError("exact, MP2, and MR CSV files describe different models")
    if (len(mp2["alpha"]) != len(mr["alpha"])
            or not np.allclose(mp2["alpha"], mr["alpha"], atol=1e-12, rtol=0)):
        raise ValueError("MR and MP2 scans must use the same alpha grid")
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (energy_ax, density_ax) = plt.subplots(
        2, 1, figsize=(5.5, 6.5), sharex=True,
        gridspec_kw={"height_ratios": (3, 2)}, constrained_layout=True,
    )
    energy_ax.plot(exact_alpha, exact_energy, color="tab:purple",
                   linewidth=1.5, marker="s", markerfacecolor="none",
                   markersize=4, label="Exact ED")
    energy_ax.plot(mp2["alpha"], mp2["cs_energy"], linestyle="--",
                   color="tab:blue", linewidth=1.3, label="CS-HF")
    energy_ax.plot(mp2["alpha"], mp2["cs_mp2_energy"], linestyle=":",
                   color="tab:green", linewidth=1.6, marker="^",
                   markerfacecolor="none", markersize=4.2, label="CS-MP2")
    energy_ax.plot(mp2["alpha"], mp2["lf_energy"], color="tab:orange",
                   linewidth=1.3, marker="o", markerfacecolor="none",
                   markersize=4.2, label="LF-HF")
    energy_ax.plot(mp2["alpha"], mp2["lf_mp2_energy"], color="tab:red",
                   linewidth=1.6, marker="D", markerfacecolor="none",
                   markersize=3.8, label="LF-MP2")
    energy_ax.plot(mr["alpha"], mr["mr_energy"], color="tab:brown",
                   linewidth=1.7, marker="v", markersize=4.2,
                   label="MR-LF NOCI (fixed frames)")
    energy_ax.set_ylabel(r"$E$")
    energy_ax.set_title(
        rf"$L={mp2_meta['L']},\ N_e=1,\ \omega={float(mp2_meta['omega']):g}$"
    )
    energy_ax.legend(frameon=False, ncol=2, fontsize=7.7)
    energy_ax.grid(alpha=0.2)

    density_ax.plot(mp2["alpha"], mp2["cs_density_imbalance"],
                    linestyle="--", color="tab:blue", linewidth=1.2,
                    label="CS-HF")
    density_ax.plot(mp2["alpha"], mp2["lf_density_imbalance"],
                    color="tab:red", linewidth=1.4, marker="o",
                    markerfacecolor="none", markersize=4.2, label="LF-HF")
    density_ax.plot(mr["alpha"], mr["mr_density_imbalance"],
                    color="tab:brown", linewidth=1.3, marker="v",
                    markersize=4.2, label="MR-LF NOCI")
    density_ax.set_xlabel(r"$\alpha=g^2/\omega$")
    density_ax.set_ylabel(r"$\max_i n_i-\min_i n_i$")
    density_ax.set_ylim(bottom=-0.02)
    density_ax.legend(frameon=False, fontsize=8)
    density_ax.grid(alpha=0.2)

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    saved_png, saved_pdf = plot_mr()
    print(f"Saved {saved_png}")
    print(f"Saved {saved_pdf}")
