"""Run deliberately cutoff-limited exact probes for Fig. 2d.

These calculations use the paper's uncentered electron-phonon coupling.  A
record is not a converged Fig. 2d datum unless both residual and cutoff flags
say so; the default strong-coupling probes are expected to remain cutoff
unconverged on the current machine.
"""

import csv
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from src import my_direct_ep as myep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "fig2d_exact_probe.csv"
CSV_COLUMNS = (
    "alpha",
    "g",
    "omega",
    "L",
    "nelec",
    "U",
    "Nmax",
    "D",
    "energy",
    "residual",
    "delta_E",
    "residual_ok",
    "cutoff_converged",
    "convention",
)


def run_fig2d_probe(
    alpha_values: Iterable[float] = (0.0, 2.0, 4.0),
    omega_values: Iterable[float] = (0.5, 5.0),
    nmax_values: Iterable[int] = (2, 4, 6, 8),
    csv_path: str | Path | None = DEFAULT_CSV_PATH,
    residual_tolerance: float = 1e-8,
    cutoff_tolerance: float = 1e-6,
    required_consecutive: int = 2,
) -> list[dict[str, object]]:
    """Evaluate and optionally persist the cutoff-limited Fig. 2d probes."""
    L = 4
    nelec = (2, 2)
    interaction = 4.0
    hopping = myep.electron_ring_hopping(L, -1.0)
    nmax_values = tuple(nmax_values)

    csv_file = None
    writer = None
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        csv_file.flush()

    records: list[dict[str, object]] = []
    try:
        for omega_value in omega_values:
            omega = float(omega_value)
            hpp = omega * np.eye(L)
            for alpha_value in alpha_values:
                alpha = float(alpha_value)
                g = float(myep.alpha_to_g(alpha, omega))
                cutoffs = (0,) if alpha == 0.0 else nmax_values
                previous_energy = None
                previous_residual_ok = False
                consecutive_small_changes = 0

                for nmax in cutoffs:
                    dimension = int(np.prod(myep.make_shape(L, nelec, nmax)))
                    energy, _, residual = myep.kernel(
                        hopping,
                        interaction,
                        g,
                        hpp,
                        L,
                        nelec,
                        nmax,
                        tol=1e-12,
                        max_cycle=200,
                        max_space=30,
                        tol_residual=residual_tolerance,
                        lindep=1e-18,
                    )
                    residual_ok = residual < residual_tolerance
                    delta_energy = (
                        None
                        if previous_energy is None
                        else abs(energy - previous_energy)
                    )

                    if (
                        delta_energy is not None
                        and delta_energy < cutoff_tolerance
                        and residual_ok
                        and previous_residual_ok
                    ):
                        consecutive_small_changes += 1
                    else:
                        consecutive_small_changes = 0
                    cutoff_converged = (
                        residual_ok
                        if alpha == 0.0
                        else consecutive_small_changes >= required_consecutive
                    )

                    record: dict[str, object] = {
                        "alpha": alpha,
                        "g": g,
                        "omega": omega,
                        "L": L,
                        "nelec": str(nelec),
                        "U": interaction,
                        "Nmax": nmax,
                        "D": dimension,
                        "energy": energy,
                        "residual": residual,
                        "delta_E": delta_energy,
                        "residual_ok": residual_ok,
                        "cutoff_converged": cutoff_converged,
                        "convention": "paper_uncentered_probe",
                    }
                    records.append(record)
                    print(
                        f"omega={omega:g} alpha={alpha:g} Nmax={nmax} "
                        f"D={dimension} E={energy:.12f} residual={residual:.2e} "
                        f"delta_E={delta_energy} cutoff_converged={cutoff_converged}",
                        flush=True,
                    )

                    if writer is not None and csv_file is not None:
                        writer.writerow(record)
                        csv_file.flush()

                    previous_energy = energy
                    previous_residual_ok = residual_ok
    finally:
        if csv_file is not None:
            csv_file.close()

    return records


if __name__ == "__main__":
    run_fig2d_probe()
