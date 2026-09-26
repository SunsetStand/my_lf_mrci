"""Plot MR-LF energy and overlap-rank convergence at the worst alpha."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.fig2b_mr_convergence import DEFAULT_CSV_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PNG_PATH = PROJECT_ROOT / "figures" / "fig2b_mr_worst_convergence.png"
DEFAULT_PDF_PATH = PROJECT_ROOT / "figures" / "fig2b_mr_worst_convergence.pdf"


def read_convergence_csv(csv_path: str | Path = DEFAULT_CSV_PATH):
    """Read the ordered frame-addition record and its model metadata."""
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("MR convergence CSV is empty")
    required = {
        "alpha", "omega", "L", "exact_energy", "step", "phase",
        "selected_theta", "n_frames", "rank", "energy",
        "energy_cut_loose", "rank_cut_loose",
        "energy_cut_tight", "rank_cut_tight",
        "overlap_cut", "overlap_cut_loose", "overlap_cut_tight",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"MR convergence CSV is missing columns: {sorted(missing)}")
    rows.sort(key=lambda row: int(row["step"]))
    alpha = float(rows[0]["alpha"])
    exact = float(rows[0]["exact_energy"])
    length = int(rows[0]["L"])
    omega = float(rows[0]["omega"])
    if any(
        not np.isclose(float(row["alpha"]), alpha)
        or not np.isclose(float(row["exact_energy"]), exact)
        or int(row["L"]) != length
        or not np.isclose(float(row["omega"]), omega)
        for row in rows
    ):
        raise ValueError("MR convergence CSV mixes models or target points")
    data = {
        "n_frames": np.array([int(row["n_frames"]) for row in rows]),
        "rank": np.array([int(row["rank"]) for row in rows]),
        "energy": np.array([float(row["energy"]) for row in rows]),
        "energy_cut_loose": np.array(
            [float(row["energy_cut_loose"]) for row in rows]
        ),
        "rank_cut_loose": np.array(
            [int(row["rank_cut_loose"]) for row in rows]
        ),
        "energy_cut_tight": np.array(
            [float(row["energy_cut_tight"]) for row in rows]
        ),
        "rank_cut_tight": np.array(
            [int(row["rank_cut_tight"]) for row in rows]
        ),
        "phase": [row["phase"] for row in rows],
        "selected_theta": [
            None if row["selected_theta"] == "" else float(row["selected_theta"])
            for row in rows
        ],
    }
    if (not np.all(np.isfinite(data["energy"]))
            or np.any(np.diff(data["n_frames"]) <= 0)):
        raise ValueError("MR convergence CSV has invalid energies or frame counts")
    return data, {
        "alpha": alpha, "exact_energy": exact, "L": length,
        "omega": omega,
        "overlap_cut": float(rows[0]["overlap_cut"]),
        "overlap_cut_loose": float(rows[0]["overlap_cut_loose"]),
        "overlap_cut_tight": float(rows[0]["overlap_cut_tight"]),
    }


def plot_convergence(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    png_path: str | Path = DEFAULT_PNG_PATH,
    pdf_path: str | Path = DEFAULT_PDF_PATH,
) -> tuple[Path, Path]:
    """Plot energy against exact ED and retained rank against nominal size."""
    data, meta = read_convergence_csv(csv_path)
    frames = data["n_frames"]
    energies = data["energy"]
    rank = data["rank"]
    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (energy_ax, rank_ax) = plt.subplots(
        2, 1, figsize=(6.2, 6.4), sharex=True,
        gridspec_kw={"height_ratios": (3, 2)}, constrained_layout=True,
    )
    energy_ax.axhline(
        meta["exact_energy"], color="tab:purple", linestyle="--",
        linewidth=1.4, label="Converged exact ED",
    )
    energy_ax.fill_between(
        frames, data["energy_cut_loose"], data["energy_cut_tight"],
        color="tab:brown", alpha=0.17,
        label="Overlap-cut sensitivity",
    )
    energy_ax.plot(
        frames, energies, color="tab:brown", marker="o",
        linewidth=1.6, markersize=4.5, label="MR-LF NOCI",
    )
    zoom = frames >= 4
    if np.count_nonzero(zoom) >= 2:
        inset = energy_ax.inset_axes((0.54, 0.34, 0.42, 0.31))
        inset.fill_between(
            frames[zoom],
            (data["energy_cut_loose"] - meta["exact_energy"])[zoom],
            (data["energy_cut_tight"] - meta["exact_energy"])[zoom],
            color="tab:brown", alpha=0.17,
        )
        inset.semilogy(
            frames[zoom], (energies - meta["exact_energy"])[zoom],
            color="tab:brown", marker="o", markersize=2.5,
            linewidth=1.0,
        )
        inset.set_title(r"$E-E_{\mathrm{ED}}$ for $K\geq4$", fontsize=7)
        inset.tick_params(labelsize=6)
        inset.grid(alpha=0.2)
    energy_ax.set_ylabel(r"$E$")
    energy_ax.set_title(
        rf"$\alpha={meta['alpha']:g},\ L={meta['L']},\ \omega={meta['omega']:g}$"
        "\nFixed displacement pool; one complete translation orbit per step"
    )
    energy_ax.legend(frameon=False, fontsize=8.5)
    energy_ax.grid(alpha=0.2)

    rank_ax.fill_between(
        frames, data["rank_cut_loose"], data["rank_cut_tight"],
        color="tab:blue", alpha=0.17,
        label="Overlap-cut sensitivity",
    )
    rank_ax.plot(
        frames, rank, color="tab:blue", marker="s",
        linewidth=1.5, markersize=4.3, label="Retained overlap rank",
    )
    rank_ax.plot(
        frames, meta["L"] * frames, color="0.5", linestyle=":",
        linewidth=1.2, label=r"Nominal $KL$",
    )
    for x, y, theta in zip(frames, rank, data["selected_theta"]):
        if theta is not None:
            rank_ax.annotate(
                rf"$\theta={theta:g}$", (x, y),
                xytext=(0, -12), textcoords="offset points",
                fontsize=7, ha="center", va="top",
            )
    rank_ax.set_xlabel("Number of LF frames $K$")
    rank_ax.set_ylabel("Basis dimension")
    rank_ax.set_xticks(frames)
    rank_ax.set_ylim(bottom=0)
    rank_ax.legend(frameon=False, fontsize=8)
    rank_ax.grid(alpha=0.2)

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


if __name__ == "__main__":
    saved_png, saved_pdf = plot_convergence()
    print(f"Saved {saved_png}")
    print(f"Saved {saved_pdf}")
