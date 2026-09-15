"""Independent diagnostic of the uncentered Fig. 2d Hamiltonian.

Validation only: builds electronic signs directly from bit strings, applies
phonon ladders by slices, and uses SciPy ARPACK instead of the project solver.
Run with ``python -m scripts.audit_fig2d_cutoff`` from the repository root.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from pyscf import lib
from pyscf.fci import direct_ep
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import LinearOperator, eigsh

from src import my_direct_ep as student


def reference_parts(nmax):
    """Fixed L=4, (2,2), t=-1, U=4, omega=.5, g=sqrt(2)."""
    strings = sorted(
        sum(1 << i for i in c) for c in itertools.combinations(range(4), 2)
    )
    addresses = {s: i for i, s in enumerate(strings)}
    t = np.zeros((4, 4))
    for i in range(4):
        t[i, (i + 1) % 4] = t[(i + 1) % 4, i] = -1.0
    spin_h = np.zeros((6, 6))
    for col, bits in enumerate(strings):
        for j in range(4):
            if not (bits >> j) & 1:
                continue
            removed = bits ^ (1 << j)
            sign_j = (-1) ** (bits & ((1 << j) - 1)).bit_count()
            for i in range(4):
                if (removed >> i) & 1:
                    continue
                sign_i = (-1) ** (removed & ((1 << i) - 1)).bit_count()
                row = addresses[removed | (1 << i)]
                spin_h[row, col] += t[i, j] * sign_i * sign_j
    kinetic = np.kron(spin_h, np.eye(6)) + np.kron(np.eye(6), spin_h)
    occupation = np.array(
        [
            [((a >> i) & 1) + ((b >> i) & 1) for i in range(4)]
            for a in strings
            for b in strings
        ]
    )
    interaction = np.array(
        [4.0 * (a & b).bit_count() for a in strings for b in strings]
    )
    d = nmax + 1
    shape = (36,) + (d,) * 4
    ph_diag = np.zeros((d,) * 4)
    for i in range(4):
        axes = [1] * 4
        axes[i] = d
        ph_diag += 0.5 * np.arange(d).reshape(axes)
    diagonal = interaction[:, None] + ph_diag.reshape(1, -1)

    def force(vector, centered=False, background=False):
        psi = vector.reshape(shape)
        result = np.zeros(shape)
        for i in range(4):
            lo, hi = [slice(None)] * 5, [slice(None)] * 5
            lo[i + 1], hi[i + 1] = slice(0, -1), slice(1, None)
            axes = [1] * 5
            axes[i + 1] = nmax
            density = np.ones(36) if background else occupation[:, i] - int(centered)
            weight = (
                np.sqrt(2)
                * density.reshape(36, 1, 1, 1, 1)
                * np.sqrt(np.arange(1, d)).reshape(axes)
            )
            result[tuple(hi)] += weight * psi[tuple(lo)]
            result[tuple(lo)] += weight * psi[tuple(hi)]
        return result.reshape(-1)

    def hop(vector):
        psi = vector.reshape(36, -1)
        out = kinetic @ psi + diagonal * psi
        out += force(vector).reshape(36, -1)
        return out.reshape(-1)

    return t, kinetic, interaction, ph_diag, force, hop


def audit(nmax):
    lib.num_threads(1)
    start = time.monotonic()
    t, kinetic, interaction, ph_diag, force, hop = reference_parts(nmax)
    shape = (6, 6) + (nmax + 1,) * 4
    dimension = int(np.prod(shape))
    rng = np.random.default_rng(20260915)
    vec = rng.standard_normal(dimension)
    vec /= np.linalg.norm(vec)
    psi = vec.reshape(shape)
    common = (4, (2, 2), nmax)
    errors = {}
    references = {
        "kinetic": (kinetic @ vec.reshape(36, -1)).reshape(-1),
        "hubbard": (interaction[:, None] * vec.reshape(36, -1)).reshape(-1),
        "phonon": (ph_diag.reshape(1, -1) * vec.reshape(36, -1)).reshape(-1),
        "ep_uncentered": force(vec),
    }
    calls = {
        "kinetic": lambda: student.contract_1e(t, psi, *common),
        "hubbard": lambda: student.contract_2e_hubbard(4.0, psi, *common),
        "phonon": lambda: student.contract_pp(0.5 * np.eye(4), psi, *common),
        "ep_uncentered": lambda: student.contract_ep_paper(np.sqrt(2), psi, *common),
    }
    for name, call in calls.items():
        errors[name] = float(np.linalg.norm(call().reshape(-1) - references[name]))
    del references
    reference = hop(vec)
    pyscf_uncentered = direct_ep.contract_all(
        t, 4.0, np.sqrt(2), 0.5 * np.eye(4), vec, *common
    )
    # Restore the uniform force, not an energy shift: identical finite basis.
    pyscf_uncentered += force(vec, background=True)
    errors["pyscf_uncentered_all"] = float(np.linalg.norm(pyscf_uncentered - reference))
    errors["student_all"] = float(
        np.linalg.norm(
            student.contract_all(
                t, 4.0, np.sqrt(2), 0.5 * np.eye(4), psi, *common
            ).reshape(-1)
            - reference
        )
    )
    errors["hdiag"] = float(
        np.max(
            np.abs(
                student.make_hdiag(t, 4.0, np.sqrt(2), 0.5 * np.eye(4), *common)
                - (
                    np.diag(kinetic)[:, None]
                    + interaction[:, None]
                    + ph_diag.reshape(1, -1)
                ).reshape(-1)
            )
        )
    )
    other = rng.standard_normal(dimension)
    other /= np.linalg.norm(other)
    errors["hermiticity_bilinear"] = float(abs(other @ reference - hop(other) @ vec))
    print("operator_errors", errors, flush=True)
    assert max(errors.values()) < 1e-11
    del other, reference, pyscf_uncentered

    atomic_rows = []
    for cutoff in (8, 12, 16, 24, 32, 48, 64):
        local = [
            float(
                eigh_tridiagonal(
                    0.5 * np.arange(cutoff + 1),
                    np.sqrt(2) * n * np.sqrt(np.arange(1, cutoff + 1)),
                    select="i",
                    select_range=(0, 0),
                )[0][0]
            )
            for n in range(3)
        ]
        # Four electrons: 1111, 2110 or 2200, all allowed in (2,2).
        atomic = min(4 * local[1], 4 + local[2] + 2 * local[1], 8 + 2 * local[2])
        atomic_rows.append(
            {
                "Nmax": cutoff,
                "local_energies": local,
                "atomic_energy": atomic,
                "atomic_error": atomic + 24.0,
                "full_energy_lower_bound": atomic + np.linalg.eigvalsh(kinetic)[0],
            }
        )
    print("atomic_bounds", atomic_rows, flush=True)
    operator = LinearOperator((dimension, dimension), matvec=hop, dtype=np.float64)
    energies, vectors = eigsh(operator, k=1, which="SA", v0=vec, ncv=24, tol=1e-11)
    energy, ground = float(energies[0]), vectors[:, 0]
    independent_residual = float(np.linalg.norm(hop(ground) - energy * ground))
    student_residual = float(
        np.linalg.norm(
            student.contract_all(
                t, 4.0, np.sqrt(2), 0.5 * np.eye(4), ground.reshape(shape), *common
            ).reshape(-1)
            - energy * ground
        )
    )
    result = {
        "L": 4,
        "nelec": [2, 2],
        "U": 4.0,
        "omega": 0.5,
        "alpha": 4.0,
        "g": float(np.sqrt(2)),
        "Nmax": nmax,
        "D": dimension,
        "convention": "paper_uncentered",
        "solver": "scipy.eigsh",
        "energy": energy,
        "independent_residual": independent_residual,
        "student_residual": student_residual,
        "operator_errors": errors,
        "atomic_bounds": atomic_rows,
        "elapsed_seconds": time.monotonic() - start,
    }
    print(json.dumps(result, indent=2), flush=True)
    assert independent_residual < 1e-8 and student_residual < 1e-8
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nmax", type=int, default=16)
    args = parser.parse_args()
    output = (
        Path(__file__).resolve().parents[1]
        / "data"
        / f"fig2d_bug_audit_nmax{args.nmax}.json"
    )
    result = audit(args.nmax)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
