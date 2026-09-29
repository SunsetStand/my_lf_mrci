"""Finish the interrupted Fig. 2d alpha=4 cutoff sequence without repeating it.

The trial vector is a superposition of all paired-electron determinants, each
with the conditional coherent displacement of the centered local oscillator.
The production Hamiltonian and eigensolver remain in ``fig2d_exact_scan``.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from pyscf import lib

from scripts.fig2d_exact_scan import COLUMNS, ROOT, solve_centered
from src import my_direct_ep as ep


def coherent_pair_guess(alpha, omega, nmax):
    """Return a normalized paired-electron, conditional-coherent trial state."""
    g_over_omega = float(ep.alpha_to_g(alpha, omega) / omega)
    strings, _ = ep.make_electron_basis(4, 2)
    state = np.zeros(ep.make_shape(4, (2, 2), nmax), dtype=np.float64)
    for address, string in enumerate(strings):
        vectors = []
        for site in range(4):
            occupation = 2 if int(string) & (1 << site) else 0
            eta = -g_over_omega * (occupation - 1)
            coherent = np.empty(nmax + 1)
            coherent[0] = np.exp(-eta * eta / 2)
            for n in range(1, nmax + 1):
                coherent[n] = coherent[n - 1] * eta / np.sqrt(n)
            vectors.append(coherent)
        state[address, address] = np.einsum(
            "i,j,k,l->ijkl", *vectors, optimize=True
        )
    state /= np.linalg.norm(state)
    return state


def finish_anchor(source, output, checkpoint_dir, *, nmax=36,
                  max_space=16, max_memory=1000):
    """Solve one new cutoff and validate it against the saved alpha=4 sequence."""
    with Path(source).open(newline="", encoding="utf-8") as handle:
        previous_rows = list(csv.DictReader(handle))
    if len(previous_rows) < 3:
        raise ValueError("the saved alpha=4 sequence is too short")
    older, latest = previous_rows[-2:]
    if (any(float(row["alpha"]) != 4 or float(row["omega"]) != .5
            for row in previous_rows)
            or int(latest["Nmax"]) >= nmax
            or int(older["Nmax"]) >= int(latest["Nmax"])):
        raise ValueError("saved sequence is not the expected alpha=4 anchor")
    cutoff_tolerance = float(latest["cutoff_tolerance"])
    residual_tolerance = float(latest["residual_tolerance"])
    if (float(latest["delta_E"]) >= cutoff_tolerance
            or float(latest["residual"]) >= residual_tolerance):
        raise ValueError("saved anchor has not reached its first small difference")

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    old_nmax = int(latest["Nmax"])
    old_centered = float(latest["energy_centered"])
    old_checkpoint = checkpoint_dir / f"omega0.5_alpha4_Nmax{old_nmax}.npy"
    if not old_checkpoint.exists():
        print(f"REBUILD omega=0.5 alpha=4 Nmax={old_nmax} "
              "for a persistent wavefunction checkpoint", flush=True)
        old_result, old_state = solve_centered(
            4.0, .5, old_nmax,
            previous=coherent_pair_guess(4.0, .5, old_nmax),
            residual_tolerance=residual_tolerance,
            max_space=max_space, max_memory=max_memory,
            include_vacuum_seed=False,
        )
        if (not old_result["solver_converged"]
                or old_result["residual"] >= residual_tolerance
                or abs(old_result["energy_centered"] - old_centered) > 1e-7):
            raise RuntimeError("rebuilt checkpoint disagrees with saved energy")
        np.save(old_checkpoint, old_state)
        del old_state
        print(f"REBUILT E={old_result['energy_centered']-16:.12f} "
              f"residual={old_result['residual']:.3e}", flush=True)
    previous_state = np.load(old_checkpoint, mmap_mode="r")
    if (previous_state.shape != ep.make_shape(4, (2, 2), old_nmax)
            or not np.all(np.isfinite(previous_state))):
        raise ValueError("stored wavefunction checkpoint is invalid")
    print(f"START omega=0.5 alpha=4 Nmax={nmax} "
          f"D={36 * (nmax + 1)**4}", flush=True)
    result, state = solve_centered(
        4.0, .5, nmax, previous=previous_state,
        residual_tolerance=residual_tolerance,
        max_space=max_space, max_memory=max_memory,
        include_vacuum_seed=False,
    )
    centered = result["energy_centered"]
    delta = abs(centered - old_centered)
    if centered > old_centered + 1e-8:
        raise RuntimeError("the larger Fock space raised the ground energy")
    okay = result["solver_converged"] and result["residual"] < residual_tolerance
    converged = okay and delta < cutoff_tolerance
    row = {
        "alpha": 4.0, "g": float(ep.alpha_to_g(4.0, .5)),
        "omega": .5, "L": 4, "neleca": 2, "nelecb": 2,
        "U": 4.0, "t": -1.0, "Nmax": nmax,
        "D": 36 * (nmax + 1)**4, **result,
        "paper_energy_shift": -16.0,
        "energy": centered - 16.0,
        "delta_E": delta, "cutoff_converged": converged,
        "cutoff_tolerance": cutoff_tolerance,
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
    np.save(checkpoint_dir / f"omega0.5_alpha4_Nmax{nmax}.npy", state)
    print(f"RESULT E={row['energy']:.12f} residual={row['residual']:.3e} "
          f"delta={delta:.3e} cutoff_converged={converged}", flush=True)
    if not converged:
        raise RuntimeError("the new cutoff has not passed both convergence checks")
    return row


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "data/fig2d_exact_anchor_convergence.csv")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/fig2d_exact_anchor_completion.csv")
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / ".fig2d_checkpoints")
    parser.add_argument("--nmax", type=int, default=36)
    parser.add_argument("--max-space", type=int, default=16)
    parser.add_argument("--max-memory", type=float, default=1000)
    args = parser.parse_args()
    lib.num_threads(1)
    finish_anchor(args.source, args.output, args.checkpoint_dir,
                  nmax=args.nmax, max_space=args.max_space,
                  max_memory=args.max_memory)
