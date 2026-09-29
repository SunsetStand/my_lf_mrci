"""Resume one interrupted Fig. 2d cutoff from a saved ED wavefunction."""
import argparse
import csv
from pathlib import Path

import numpy as np
from pyscf import lib

from scripts.fig2d_exact_scan import COLUMNS, solve_centered
from src import my_direct_ep as ep


def resume_cutoff(source, checkpoint, output, *, alpha, omega, nmax,
                  max_space=16, max_memory=1000):
    """Solve the next larger cutoff and verify two successive small changes."""
    with Path(source).open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)
                if float(row["alpha"]) == alpha and float(row["omega"]) == omega]
    if len(rows) < 2:
        raise ValueError("saved point needs two preceding cutoff rows")
    old = rows[-1]
    old_nmax = int(old["Nmax"])
    if nmax <= old_nmax or old["cutoff_converged"] == "True":
        raise ValueError("requested cutoff must extend an unconverged point")
    tolerance = float(old["cutoff_tolerance"])
    residual_tolerance = float(old["residual_tolerance"])
    if (float(old["delta_E"]) >= tolerance
            or float(old["residual"]) >= residual_tolerance):
        raise ValueError("previous cutoff has not recorded the first small change")
    previous = np.load(checkpoint, mmap_mode="r")
    if (previous.shape != ep.make_shape(4, (2, 2), old_nmax)
            or not np.all(np.isfinite(previous))):
        raise ValueError("checkpoint shape or values disagree with the CSV")
    print(f"START omega={omega:g} alpha={alpha:g} Nmax={nmax} "
          f"D={36*(nmax+1)**4}", flush=True)
    result, state = solve_centered(
        alpha, omega, nmax, previous=previous,
        residual_tolerance=residual_tolerance,
        max_space=max_space, max_memory=max_memory,
        include_vacuum_seed=False,
    )
    energy = result["energy_centered"]
    old_energy = float(old["energy_centered"])
    delta = abs(energy - old_energy)
    if energy > old_energy + 1e-8:
        raise RuntimeError("larger Fock space raised the ground energy")
    converged = (result["solver_converged"]
                 and result["residual"] < residual_tolerance
                 and delta < tolerance)
    row = {
        "alpha": alpha, "g": float(ep.alpha_to_g(alpha, omega)),
        "omega": omega, "L": 4, "neleca": 2, "nelecb": 2,
        "U": 4.0, "t": -1.0, "Nmax": nmax,
        "D": 36*(nmax+1)**4, **result,
        "paper_energy_shift": -4*alpha,
        "energy": energy-4*alpha,
        "delta_E": delta, "cutoff_converged": converged,
        "cutoff_tolerance": tolerance,
        "residual_tolerance": residual_tolerance,
        "required_consecutive": 2,
        "convention": "centered_solve_paper_energy",
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    saved_state = Path(checkpoint).with_name(
        f"omega{omega:g}_alpha{alpha:g}_Nmax{nmax}.npy")
    np.save(saved_state, state)
    print(f"RESULT E={row['energy']:.12f} residual={row['residual']:.3e} "
          f"delta={delta:.3e} cutoff_converged={converged}", flush=True)
    if not converged:
        raise RuntimeError("cutoff or eigensolver criterion was not met")
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--omega", type=float, required=True)
    parser.add_argument("--nmax", type=int, required=True)
    parser.add_argument("--max-space", type=int, default=16)
    parser.add_argument("--max-memory", type=float, default=1000)
    args = parser.parse_args()
    lib.num_threads(1)
    resume_cutoff(args.source, args.checkpoint, args.output,
                  alpha=args.alpha, omega=args.omega, nmax=args.nmax,
                  max_space=args.max_space, max_memory=args.max_memory)
