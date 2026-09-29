"""Compare the pre-break cross-alpha MR-LF frame scan with Fig. 2b curves."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.fig2b_mr_broken_seed_scan import DEFAULT_CSV_PATH, FRAME_RECIPE
from scripts.plot_fig2b_exact import DEFAULT_CSV_PATH as EXACT_CSV_PATH
from scripts.plot_fig2b_exact import read_exact_csv
from scripts.plot_fig2b_mp2 import DEFAULT_MP2_CSV_PATH, read_mp2_csv
from scripts.plot_fig2b_mr import DEFAULT_MR_CSV_PATH, read_mr_csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PNG_PATH = PROJECT_ROOT / "figures" / "fig2b_mr_broken_seed.png"
DEFAULT_PDF_PATH = PROJECT_ROOT / "figures" / "fig2b_mr_broken_seed.pdf"


def read_broken_seed_csv(csv_path: str | Path = DEFAULT_CSV_PATH):
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("broken-seed scan CSV is empty")
    required = {
        "alpha", "omega", "L", "t", "coupling_convention",
        "source_alpha", "frame_recipe", "energy_base", "energy_augmented",
        "rank_augmented", "density_imbalance",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"broken-seed CSV is missing columns: {sorted(missing)}")
    metadata = {key: rows[0][key] for key in (
        "omega", "L", "t", "coupling_convention", "source_alpha", "frame_recipe",
    )}
    if any(any(row[key] != value for key, value in metadata.items())
           for row in rows):
        raise ValueError("broken-seed CSV mixes model conventions or source frames")
    if metadata["frame_recipe"] != FRAME_RECIPE:
        raise ValueError("broken-seed CSV has an unknown frame recipe")
    rows.sort(key=lambda row: float(row["alpha"]))
    data = {
        "alpha": np.array([float(row["alpha"]) for row in rows]),
        "energy_base": np.array([float(row["energy_base"]) for row in rows]),
        "energy_augmented": np.array(
            [float(row["energy_augmented"]) for row in rows]
        ),
        "rank_augmented": np.array(
            [int(row["rank_augmented"]) for row in rows], dtype=np.int64
        ),
        "density_imbalance": np.array(
            [float(row["density_imbalance"]) for row in rows]
        ),
    }
    if any(not np.all(np.isfinite(values)) for values in data.values()):
        raise ValueError("broken-seed CSV contains nonfinite values")
    if np.any(np.diff(data["alpha"]) <= 0):
        raise ValueError("broken-seed alpha grid must be unique")
    return data, metadata


def plot_broken_seed(
    new_csv_path: str | Path = DEFAULT_CSV_PATH,
    mr_csv_path: str | Path = DEFAULT_MR_CSV_PATH,
    mp2_csv_path: str | Path = DEFAULT_MP2_CSV_PATH,
    exact_csv_path: str | Path = EXACT_CSV_PATH,
    png_path: str | Path = DEFAULT_PNG_PATH,
    pdf_path: str | Path = DEFAULT_PDF_PATH,
) -> tuple[Path, Path]:
    new, new_meta = read_broken_seed_csv(new_csv_path)
    mr, mr_meta = read_mr_csv(mr_csv_path)
    mp2, mp2_meta = read_mp2_csv(mp2_csv_path)
    exact_alpha, exact_energy, exact_meta = read_exact_csv(exact_csv_path)
    if (
        int(new_meta["L"]) != int(mr_meta["L"])
        or int(new_meta["L"]) != int(mp2_meta["L"])
        or int(new_meta["L"]) != int(exact_meta["L"])
        or not np.isclose(float(new_meta["omega"]), float(mr_meta["omega"]))
        or not np.isclose(float(new_meta["omega"]), float(mp2_meta["omega"]))
        or not np.isclose(float(new_meta["omega"]), float(exact_meta["omega"]))
        or not np.isclose(float(new_meta["t"]), float(mr_meta["t"]))
        or not np.isclose(float(new_meta["t"]), float(mp2_meta["t"]))
        or new_meta["coupling_convention"] != mr_meta["coupling_convention"]
        or new_meta["coupling_convention"] != mp2_meta["coupling_convention"]
        or new_meta["coupling_convention"] != exact_meta.get("convention")
    ):
        raise ValueError("comparison CSV files describe different models")
    if (
        len(exact_alpha) != len(mr["alpha"])
        or len(exact_alpha) != len(mp2["alpha"])
        or not np.allclose(exact_alpha, mr["alpha"], rtol=0, atol=1e-12)
        or not np.allclose(exact_alpha, mp2["alpha"], rtol=0, atol=1e-12)
        or not np.allclose(
            exact_alpha[:len(new["alpha"])], new["alpha"], rtol=0, atol=1e-12
        )
    ):
        raise ValueError("comparison alpha grids do not match")
    if (
        np.any(new["alpha"] >= float(new_meta["source_alpha"]))
        or not np.allclose(
            new["energy_base"], mr["mr_energy"][:len(new["alpha"])],
            rtol=0, atol=1e-8,
        )
    ):
        raise ValueError("new scan does not match the pre-break MR baseline")

    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (energy_ax, error_ax) = plt.subplots(
        2, 1, figsize=(6.4, 7.0), sharex=False,
        gridspec_kw={"height_ratios": (3, 2)}, constrained_layout=True,
    )
    curves = (
        (exact_alpha, exact_energy, "Exact ED", "tab:purple", "-", "s"),
        (mp2["alpha"], mp2["cs_energy"], "CS-HF", "tab:blue", "--", None),
        (mp2["alpha"], mp2["cs_mp2_energy"], "CS-MP2", "tab:green", ":", "^"),
        (mp2["alpha"], mp2["lf_energy"], "LF-HF", "tab:orange", "-", "o"),
        (mp2["alpha"], mp2["lf_mp2_energy"], "LF-MP2", "tab:red", "-", "D"),
        (mr["alpha"], mr["mr_energy"], "MR-LF original", "tab:brown", "-", "v"),
        (new["alpha"], new["energy_augmented"],
         "MR-LF + broken orbit", "tab:cyan", "-", "P"),
    )
    for alpha, energy, label, color, style, marker in curves:
        energy_ax.plot(
            alpha, energy, color=color, linestyle=style, linewidth=1.5,
            marker=marker, markersize=4, markerfacecolor="none",
            label=label,
        )
    energy_ax.set_ylabel(r"$E$")
    energy_ax.set_title(
        rf"$L={new_meta['L']},\ N_e=1,\ \omega={float(new_meta['omega']):g}$"
        + rf"; broken seed at $\alpha={float(new_meta['source_alpha']):g}$"
    )
    energy_ax.grid(alpha=0.2)
    energy_ax.legend(frameon=False, ncol=2, fontsize=8)

    count = len(new["alpha"])
    mask = new["alpha"] > 0
    x = new["alpha"][mask]
    ref = exact_energy[:count][mask]
    error_curves = (
        (mp2["lf_energy"][:count][mask], "LF-HF", "tab:orange", "o"),
        (mp2["lf_mp2_energy"][:count][mask], "LF-MP2", "tab:red", "D"),
        (mr["mr_energy"][:count][mask], "MR-LF original", "tab:brown", "v"),
        (new["energy_augmented"][mask],
         "MR-LF + broken orbit", "tab:cyan", "P"),
    )
    for energy, label, color, marker in error_curves:
        error_ax.semilogy(
            x, np.abs(energy - ref), color=color, linewidth=1.5,
            marker=marker, markersize=4, markerfacecolor="none", label=label,
        )
    error_ax.set_xlabel(r"$\alpha=g^2/\omega$")
    error_ax.set_ylabel(r"$|E-E_{\mathrm{ED}}|$")
    error_ax.set_title("Before LF-HF symmetry breaking")
    error_ax.grid(alpha=0.2)
    error_ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    png, pdf = plot_broken_seed()
    print(f"Saved {png}")
    print(f"Saved {pdf}")
