"""Run the centered-basis cutoff sequence for the difficult Fig. 2d point."""

import argparse
import csv
from pathlib import Path

import numpy as np

from src import my_direct_ep as myep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "fig2d_centered_omega0p5_alpha4_convergence.csv"
)
FIELDNAMES = (
    "alpha",
    "g",
    "omega",
    "L",
    "nelec",
    "U",
    "Nmax",
    "D",
    "energy_centered",
    "paper_energy_shift",
    "energy_paper",
    "residual",
    "delta_E",
    "residual_ok",
    "cutoff_converged",
    "coupling_convention",
)


def run_sequence(nmax_values, output_path):
    """Run and save one centered cutoff sequence, flushing after every point."""
    nsite = 4
    nelec = (2, 2)
    interaction = 4.0
    omega = 0.5
    alpha = 4.0
    coupling = float(myep.alpha_to_g(alpha, omega))
    hopping = myep.electron_ring_hopping(nsite, -1.0)
    hpp = omega * np.eye(nsite)
    paper_energy_shift = -alpha * sum(nelec) ** 2 / nsite
    residual_tolerance = 1e-8
    cutoff_tolerance = 1e-6
    previous_energy = None
    consecutive_small_changes = 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        csv_file.flush()

        for nmax in nmax_values:
            dimension = int(np.prod(myep.make_shape(nsite, nelec, nmax)))
            energy_centered, _, residual = myep.kernel(
                hopping,
                interaction,
                coupling,
                hpp,
                nsite,
                nelec,
                nmax,
                tol=1e-12,
                max_cycle=400,
                max_space=30,
                tol_residual=residual_tolerance,
                lindep=1e-18,
                coupling_convention="centered",
            )
            energy_paper = energy_centered + paper_energy_shift
            delta_energy = (
                None
                if previous_energy is None
                else abs(energy_centered - previous_energy)
            )
            residual_ok = residual < residual_tolerance
            if (
                delta_energy is not None
                and delta_energy < cutoff_tolerance
                and residual_ok
            ):
                consecutive_small_changes += 1
            else:
                consecutive_small_changes = 0
            cutoff_converged = consecutive_small_changes >= 2
            row = {
                "alpha": alpha,
                "g": coupling,
                "omega": omega,
                "L": nsite,
                "nelec": str(nelec),
                "U": interaction,
                "Nmax": nmax,
                "D": dimension,
                "energy_centered": energy_centered,
                "paper_energy_shift": paper_energy_shift,
                "energy_paper": energy_paper,
                "residual": residual,
                "delta_E": delta_energy,
                "residual_ok": residual_ok,
                "cutoff_converged": cutoff_converged,
                "coupling_convention": "centered",
            }
            writer.writerow(row)
            csv_file.flush()
            print(
                f"Nmax={nmax:2d} D={dimension:8d} "
                f"Ec={energy_centered:.12f} Epaper={energy_paper:.12f} "
                f"residual={residual:.2e} delta_E={delta_energy}",
                flush=True,
            )
            previous_energy = energy_centered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nmax-values", nargs="+", type=int, default=(2, 4, 6, 8, 10, 12, 14, 16)
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    run_sequence(arguments.nmax_values, arguments.output)
