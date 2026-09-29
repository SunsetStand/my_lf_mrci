"""Centered four-electron Fig. 2d ED scan using the student's contractions.

Only solver orchestration, warm starts, caching, and output live here.
Every Hamiltonian action and its diagonal come from src.my_direct_ep.
"""
import argparse
import csv
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from pyscf import lib
from scipy.sparse.linalg import LinearOperator, eigsh

from src import my_direct_ep as ep

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = (
    "alpha", "g", "omega", "L", "neleca", "nelecb", "U", "t",
    "Nmax", "D", "energy_centered", "paper_energy_shift", "energy",
    "residual", "solver", "solver_converged", "delta_E", "cutoff_converged",
    "cutoff_tolerance", "residual_tolerance", "required_consecutive",
    "convention", "seconds", "matvecs",
)


def resize_guess(vector, nmax):
    """Embed or crop a previous Fock tensor and normalize its projection."""
    shape = ep.make_shape(4, (2, 2), nmax)
    out = np.zeros(shape)
    if vector is not None:
        d = min(vector.shape[-1], nmax + 1)
        block = (slice(None), slice(None)) + (slice(0, d),) * 4
        out[block] = vector[block]
    else:
        rng = np.random.default_rng(872)
        out[:, :, 0, 0, 0, 0] = rng.normal(size=(6, 6))
    norm = np.linalg.norm(out)
    if norm == 0:
        raise ValueError("initial projection has zero norm")
    out /= norm
    return out


def solve_centered(alpha, omega, nmax, previous=None, *,
                   residual_tolerance=1e-8, max_space=40,
                   max_memory=1400, max_cycle=500,
                   include_vacuum_seed=True):
    """Run memory-limited Davidson with unchanged production contractions."""
    tmat = ep.electron_ring_hopping(4, -1.0)
    g = float(ep.alpha_to_g(alpha, omega))
    hpp = omega * np.eye(4)
    shape = ep.make_shape(4, (2, 2), nmax)
    calls = 0
    started = perf_counter()

    # The occupation table is invariant over all Davidson iterations.
    # Cache the existing function rather than reimplementing its ordering.
    cached_configs = lru_cache(maxsize=1)(ep.phonon_configs)
    with patch.object(ep, "phonon_configs", cached_configs):
        diagonal = ep.make_hdiag(tmat, 4.0, g, hpp, 4, (2, 2), nmax)
        guesses = [resize_guess(previous, nmax).ravel()]
        if previous is not None and include_vacuum_seed:
            # Keep the accurate embedded state intact. A separate electronic
            # vacuum seed allows competing symmetry sectors into the solve.
            guesses.append(resize_guess(None, nmax).ravel())
        previous = None  # Release a caller-owned trial when passed inline.

        def hop(vector):
            nonlocal calls
            calls += 1
            return ep.contract_all(
                tmat, 4.0, g, hpp, vector.reshape(shape),
                4, (2, 2), nmax, coupling_convention="centered",
            ).ravel()

        def progress(env):
            cycle = env.get("icyc", -1)
            if cycle >= 0 and cycle % 25 == 0:
                print(f"  cycle={cycle} matvecs={calls} "
                      f"elapsed={perf_counter()-started:.1f}s", flush=True)

        flags, _energies, vectors = lib.davidson1(
            lambda vs: [hop(v) for v in vs],
            guesses, lib.make_diag_precond(diagonal),
            tol=1e-12, tol_residual=residual_tolerance,
            lindep=1e-18, max_cycle=max_cycle, max_space=max_space,
            max_memory=max_memory, lessio=True, callback=progress, verbose=0,
        )
        vector = vectors[0]
        vector /= np.linalg.norm(vector)
        acted = hop(vector)
        energy = float(np.dot(vector, acted))
        residual = float(np.linalg.norm(acted - energy * vector))
        solver = "davidson"
        solver_ok = bool(flags[0])
        if not solver_ok or residual >= residual_tolerance:
            # Generic eigensolver fallback for observed Davidson stagnation;
            # the Hamiltonian action is exactly the same production function.
            print(f"  Davidson stalled at residual={residual:.3e}; "
                  "retrying with Lanczos", flush=True)
            dimension = vector.size
            ncv = min(40, max(4, int(max_memory*1e6/(8*dimension))-8))
            operator = LinearOperator((dimension, dimension), matvec=hop,
                                      dtype=np.float64)
            _evals, evecs = eigsh(operator, k=1, which="SA", v0=vector,
                                  tol=1e-12, maxiter=2000,
                                  ncv=min(ncv, dimension))
            vector = evecs[:, 0]
            vector /= np.linalg.norm(vector)
            acted = hop(vector)
            energy = float(np.dot(vector, acted))
            residual = float(np.linalg.norm(acted-energy*vector))
            solver, solver_ok = "eigsh_after_davidson", True
    return {
        "energy_centered": energy, "residual": residual, "solver": solver,
        "solver_converged": solver_ok,
        "seconds": perf_counter() - started, "matvecs": calls,
    }, vector.reshape(shape)


def run_scan(alpha_values, omega_values, nmax_values, csv_path, *,
             cutoff_tolerance=1e-6, residual_tolerance=1e-8,
             required_consecutive=2, max_memory=1400, max_space=40,
             checkpoint_dir=None):
    """Persist every cutoff; stop a point only after two small changes."""
    alpha_values = tuple(float(a) for a in alpha_values)
    omega_values = tuple(float(w) for w in omega_values)
    nmax_values = tuple(nmax_values)
    if not alpha_values or any(not np.isfinite(a) or a < 0 for a in alpha_values):
        raise ValueError("alpha grid must be finite, nonempty, and nonnegative")
    if not omega_values or any(not np.isfinite(w) or w <= 0 for w in omega_values):
        raise ValueError("omega grid must be finite, nonempty, and positive")
    if (not nmax_values or any(not isinstance(n, (int, np.integer)) or n < 0
                               for n in nmax_values)
            or any(b <= a for a, b in pairwise(nmax_values))):
        raise ValueError("phonon cutoffs must be nonnegative and strictly increasing")
    if (not np.isfinite(cutoff_tolerance) or cutoff_tolerance <= 0
            or not np.isfinite(residual_tolerance) or residual_tolerance <= 0
            or required_consecutive < 1):
        raise ValueError("convergence tolerances and required count must be positive")
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        handle.flush()
        for omega in omega_values:
            for alpha in alpha_values:
                previous = None
                previous_energy = None
                previous_ok = False
                small_changes = 0
                cutoffs = (0,) if alpha == 0 else tuple(nmax_values)
                for nmax in cutoffs:
                    print(f"START omega={omega:g} alpha={alpha:g} "
                          f"Nmax={nmax} D={36*(nmax+1)**4}", flush=True)
                    result, previous = solve_centered(
                        alpha, omega, nmax, previous,
                        residual_tolerance=residual_tolerance,
                        max_memory=max_memory, max_space=max_space,
                    )
                    ec = result["energy_centered"]
                    delta = None if previous_energy is None else abs(ec-previous_energy)
                    okay = (result["solver_converged"]
                            and result["residual"] < residual_tolerance)
                    if (delta is not None and delta < cutoff_tolerance
                            and okay and previous_ok):
                        small_changes += 1
                    else:
                        small_changes = 0
                    converged = okay and (
                        alpha == 0 or small_changes >= required_consecutive
                    )
                    record = {
                        "alpha": float(alpha), "g": float(ep.alpha_to_g(alpha, omega)),
                        "omega": float(omega), "L": 4, "neleca": 2, "nelecb": 2,
                        "U": 4.0, "t": -1.0, "Nmax": int(nmax),
                        "D": 36*(nmax+1)**4, **result,
                        "paper_energy_shift": -4.0*alpha,
                        "energy": ec-4.0*alpha, "delta_E": delta,
                        "cutoff_converged": converged,
                        "cutoff_tolerance": cutoff_tolerance,
                        "residual_tolerance": residual_tolerance,
                        "required_consecutive": required_consecutive,
                        "convention": "centered_solve_paper_energy",
                    }
                    writer.writerow(record)
                    handle.flush()
                    records.append(record)
                    print(f"RESULT E={record['energy']:.12f} "
                          f"residual={result['residual']:.3e} "
                          f"delta={delta} cutoff_converged={converged} "
                          f"seconds={result['seconds']:.1f}", flush=True)
                    if not okay:
                        raise RuntimeError("eigensolver residual criterion was not met")
                    if checkpoint_dir is not None:
                        checkpoint = Path(checkpoint_dir)
                        checkpoint.mkdir(parents=True, exist_ok=True)
                        np.save(checkpoint / f"omega{omega:g}_alpha{alpha:g}.npy",
                                previous)
                    if previous_energy is not None and ec > previous_energy + 1e-8:
                        raise RuntimeError("larger Fock space raised the ground energy")
                    previous_energy, previous_ok = ec, okay
                    if converged:
                        break
                if not records[-1]["cutoff_converged"]:
                    print("CUTOFF LIMIT reached without convergence", flush=True)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=np.linspace(0, 4, 11))
    parser.add_argument("--omegas", nargs="+", type=float, default=[0.5, 5.0])
    parser.add_argument("--nmax-values", nargs="+", type=int,
                        default=list(range(2, 37, 2)))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/fig2d_exact_convergence.csv")
    parser.add_argument("--max-memory", type=float, default=1400)
    parser.add_argument("--max-space", type=int, default=40)
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    lib.num_threads(1)
    run_scan(args.alphas, args.omegas, args.nmax_values, args.output,
             max_memory=args.max_memory, max_space=args.max_space,
             checkpoint_dir=args.checkpoint_dir)
