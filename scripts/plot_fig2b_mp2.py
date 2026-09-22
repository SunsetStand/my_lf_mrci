"""Plot exact, CS-HF/MP2, and full-matrix LF-HF/MP2 for Fig. 2b."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.fig2b_mp2_scan import DEFAULT_CSV_PATH as DEFAULT_MP2_CSV_PATH
from scripts.plot_fig2b_exact import DEFAULT_CSV_PATH as DEFAULT_EXACT_CSV_PATH
from scripts.plot_fig2b_exact import read_exact_csv
from scripts.plot_fig2b_hf import read_hf_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PNG_PATH = PROJECT_ROOT / "figures" / "fig2b_mp2.png"
DEFAULT_PDF_PATH = PROJECT_ROOT / "figures" / "fig2b_mp2.pdf"


def read_mp2_csv(
    csv_path: str | Path = DEFAULT_MP2_CSV_PATH,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read and sort the combined HF/MP2 scan."""
    data, metadata = read_hf_csv(csv_path)
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    required = {
        "lf_mp2_max_total",
        "lf_mp2_nconfig",
        "lf_mp2_pure_correction",
        "lf_mp2_single_correction",
        "lf_mp2_correction",
        "lf_mp2_energy",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"MP2 scan CSV is missing columns: {sorted(missing)}")
    rows.sort(key=lambda row: float(row["alpha"]))
    data.update(
        {
            "lf_mp2_max_total": np.array(
                [int(row["lf_mp2_max_total"]) for row in rows], dtype=np.int64
            ),
            "lf_mp2_nconfig": np.array(
                [int(row["lf_mp2_nconfig"]) for row in rows], dtype=np.int64
            ),
            "lf_mp2_pure_correction": np.array(
                [float(row["lf_mp2_pure_correction"]) for row in rows]
            ),
            "lf_mp2_single_correction": np.array(
                [float(row["lf_mp2_single_correction"]) for row in rows]
            ),
            "lf_mp2_correction": np.array(
                [float(row["lf_mp2_correction"]) for row in rows]
            ),
            "lf_mp2_energy": np.array(
                [float(row["lf_mp2_energy"]) for row in rows]
            ),
        }
    )
    return data, metadata


def plot_mp2(
    csv_path: str | Path = DEFAULT_MP2_CSV_PATH,
    png_path: str | Path = DEFAULT_PNG_PATH,
    pdf_path: str | Path = DEFAULT_PDF_PATH,
    exact_csv_path: str | Path = DEFAULT_EXACT_CSV_PATH,
) -> tuple[Path, Path]:
    """Plot all available Fig. 2b energy curves and HF density imbalance."""
    data, metadata = read_mp2_csv(csv_path)
    exact_alpha, exact_energy, exact_metadata = read_exact_csv(exact_csv_path)
    if (
        int(exact_metadata["L"]) != int(metadata["L"])
        or not np.isclose(float(exact_metadata["omega"]), float(metadata["omega"]))
    ):
        raise ValueError("exact and MP2 CSV files describe different models")
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (energy_ax, density_ax) = plt.subplots(
        2,
        1,
        figsize=(5.4, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 2)},
        constrained_layout=True,
    )
    energy_ax.plot(
        exact_alpha,
        exact_energy,
        color="tab:purple",
        linewidth=1.5,
        marker="s",
        markerfacecolor="none",
        markersize=4.0,
        label="Exact ED",
    )
    energy_ax.plot(
        data["alpha"],
        data["cs_energy"],
        linestyle="--",
        color="tab:blue",
        linewidth=1.3,
        label="CS-HF",
    )
    energy_ax.plot(
        data["alpha"],
        data["cs_mp2_energy"],
        linestyle=":",
        color="tab:green",
        linewidth=1.6,
        marker="^",
        markerfacecolor="none",
        markersize=4.2,
        label="CS-MP2",
    )
    energy_ax.plot(
        data["alpha"],
        data["lf_energy"],
        color="tab:orange",
        linewidth=1.3,
        marker="o",
        markerfacecolor="none",
        markersize=4.2,
        label="LF-HF",
    )
    energy_ax.plot(
        data["alpha"],
        data["lf_mp2_energy"],
        color="tab:red",
        linewidth=1.6,
        marker="D",
        markerfacecolor="none",
        markersize=3.8,
        label="LF-MP2",
    )
    energy_ax.set_ylabel(r"$E$")
    energy_ax.set_title(
        rf"$L={metadata['L']},\ N_e=1,\ \omega={float(metadata['omega']):g}$"
    )
    energy_ax.legend(frameon=False, ncol=2, fontsize=8.5)
    energy_ax.grid(alpha=0.2)

    density_ax.plot(
        data["alpha"],
        data["cs_density_imbalance"],
        linestyle="--",
        color="tab:blue",
        linewidth=1.2,
        label="CS-HF",
    )
    density_ax.plot(
        data["alpha"],
        data["lf_density_imbalance"],
        color="tab:red",
        linewidth=1.4,
        marker="o",
        markerfacecolor="none",
        markersize=4.2,
        label="LF-HF",
    )
    density_ax.set_xlabel(r"$\alpha=g^2/\omega$")
    density_ax.set_ylabel(r"$\max_i n_i-\min_i n_i$")
    density_ax.set_ylim(bottom=-0.02)
    density_ax.legend(frameon=False)
    density_ax.grid(alpha=0.2)

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    saved_png, saved_pdf = plot_mp2()
    print(f"Saved {saved_png}")
    print(f"Saved {saved_pdf}")
