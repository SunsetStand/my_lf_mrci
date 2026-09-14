"""Plot the exact part of Fig. 2b from the saved alpha scan."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "fig2b_exact.csv"
DEFAULT_PNG_PATH = PROJECT_ROOT / "figures" / "fig2b_exact.png"
DEFAULT_PDF_PATH = PROJECT_ROOT / "figures" / "fig2b_exact.pdf"


def read_exact_csv(
    csv_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Read alpha, energy, and constant run metadata from an exact scan CSV."""
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError("exact scan CSV contains no data rows")

    rows.sort(key=lambda row: float(row["alpha"]))
    alpha = np.array([float(row["alpha"]) for row in rows])
    energy = np.array([float(row["energy"]) for row in rows])
    metadata = rows[0]
    return alpha, energy, metadata


def plot_exact(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    png_path: str | Path = DEFAULT_PNG_PATH,
    pdf_path: str | Path = DEFAULT_PDF_PATH,
) -> tuple[Path, Path]:
    """Create PNG and PDF plots for the exact Fig. 2b data."""
    alpha, energy, metadata = read_exact_csv(csv_path)
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    ax.plot(
        alpha,
        energy,
        color="tab:purple",
        linewidth=1.2,
        marker="o",
        markerfacecolor="none",
        markersize=4.5,
        label="exact",
    )
    ax.set_xlabel(r"$g^2/\omega$")
    ax.set_ylabel(r"$E$")
    ax.set_title(
        rf"$L={metadata['L']},\ n_{{\mathrm{{elec}}}}="
        rf"{int(metadata['neleca']) + int(metadata['nelecb'])},\ "
        rf"\omega={float(metadata['omega']):g}$"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path


if __name__ == "__main__":
    png_path, pdf_path = plot_exact()
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
