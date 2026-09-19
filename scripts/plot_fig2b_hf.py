"""Plot the CS-HF and local LF-HF parts of the Fig. 2b reproduction."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.fig2b_hf_scan import DEFAULT_CSV_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PNG_PATH = PROJECT_ROOT / "figures" / "fig2b_hf.png"
DEFAULT_PDF_PATH = PROJECT_ROOT / "figures" / "fig2b_hf.pdf"


def read_hf_csv(csv_path: str | Path = DEFAULT_CSV_PATH) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Read and sort the HF scan while preserving all four site densities."""
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError("HF scan CSV contains no data rows")
    required = {
        "alpha",
        "cs_energy",
        "lf_energy",
        "cs_density_imbalance",
        "lf_density_imbalance",
        *(f"cs_n{site}" for site in range(4)),
        *(f"lf_n{site}" for site in range(4)),
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"HF scan CSV is missing columns: {sorted(missing)}")

    rows.sort(key=lambda row: float(row["alpha"]))
    data = {
        "alpha": np.array([float(row["alpha"]) for row in rows]),
        "cs_energy": np.array([float(row["cs_energy"]) for row in rows]),
        "lf_energy": np.array([float(row["lf_energy"]) for row in rows]),
        "cs_density": np.array(
            [[float(row[f"cs_n{site}"]) for site in range(4)] for row in rows]
        ),
        "lf_density": np.array(
            [[float(row[f"lf_n{site}"]) for site in range(4)] for row in rows]
        ),
        "cs_density_imbalance": np.array(
            [float(row["cs_density_imbalance"]) for row in rows]
        ),
        "lf_density_imbalance": np.array(
            [float(row["lf_density_imbalance"]) for row in rows]
        ),
    }
    return data, rows[0]


def plot_hf(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    png_path: str | Path = DEFAULT_PNG_PATH,
    pdf_path: str | Path = DEFAULT_PDF_PATH,
) -> tuple[Path, Path]:
    """Create energy and density-imbalance panels from an HF scan CSV."""
    data, metadata = read_hf_csv(csv_path)
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (energy_ax, density_ax) = plt.subplots(
        2,
        1,
        figsize=(5.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": (3, 2)},
        constrained_layout=True,
    )
    energy_ax.plot(
        data["alpha"],
        data["cs_energy"],
        linestyle="--",
        color="tab:blue",
        linewidth=1.4,
        label="CS-HF (symmetric)",
    )
    energy_ax.plot(
        data["alpha"],
        data["lf_energy"],
        color="tab:orange",
        linewidth=1.4,
        marker="o",
        markerfacecolor="none",
        markersize=4.5,
        label="LF-HF (local)",
    )
    energy_ax.set_ylabel(r"$E$")
    energy_ax.set_title(
        rf"$L={metadata['L']},\ N_e=1,\ \omega={float(metadata['omega']):g}$"
    )
    energy_ax.legend(frameon=False)
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
        markersize=4.5,
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
    saved_png, saved_pdf = plot_hf()
    print(f"Saved {saved_png}")
    print(f"Saved {saved_pdf}")
