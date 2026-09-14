"""Scan the exact Fig. 2b energies and save each completed point."""

import csv
from pathlib import Path

import numpy as np

from src import my_direct_ep as myep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "fig2b_exact.csv"
CSV_COLUMNS = (
    "alpha",
    "g",
    "omega",
    "L",
    "neleca",
    "nelecb",
    "Nmax",
    "D",
    "energy",
    "residual",
    "convention",
)


def run_exact_scan(
    alpha_values: np.ndarray,
    nmax: int = 24,
    csv_path: str | Path | None = None,
) -> list[tuple[float, float, int, int, float, float]]:
    """Run exact calculations, optionally saving each completed point to CSV."""
    L = 4
    nelec = (1, 0)
    omega = 0.5
    U = 0.0
    t = -1.0
    Ht = myep.electron_ring_hopping(L, t)
    hpp = np.eye(L) * omega
    D = int(np.prod(myep.make_shape(L, nelec, nmax)))

    print("coupling_convention = paper_uncentered")
    print(f"L={L}, nelec={nelec}, omega={omega}, nmax={nmax}")
    print("alpha\tg\tD\tEnergy\tResidual")

    csv_file = None
    writer = None
    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        csv_file.flush()

    records = []
    try:
        for alpha_value in alpha_values:
            alpha = float(alpha_value)
            g = float(myep.alpha_to_g(alpha, omega))
            energy, _, residual = myep.kernel(Ht, U, g, hpp, L, nelec, nmax)
            record = (alpha, g, nmax, D, energy, residual)
            records.append(record)

            print(
                f"{alpha:.2f}\t{g:.6f}\t{D}\t"
                f"{energy:.12f}\t{residual:.2e}",
                flush=True,
            )

            if writer is not None and csv_file is not None:
                writer.writerow(
                    {
                        "alpha": f"{alpha:.10g}",
                        "g": f"{g:.16g}",
                        "omega": omega,
                        "L": L,
                        "neleca": nelec[0],
                        "nelecb": nelec[1],
                        "Nmax": nmax,
                        "D": D,
                        "energy": f"{energy:.16g}",
                        "residual": f"{residual:.16g}",
                        "convention": "paper_uncentered",
                    }
                )
                csv_file.flush()
    finally:
        if csv_file is not None:
            csv_file.close()

    return records


if __name__ == "__main__":
    alpha_values = np.linspace(0.0, 3.0, 16)
    run_exact_scan(alpha_values, nmax=24, csv_path=DEFAULT_CSV_PATH)
