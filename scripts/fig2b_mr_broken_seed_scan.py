"""Add the alpha=2.4 broken LF frame orbit to each pre-break Fig. 2b MR space."""

import csv
from pathlib import Path

import numpy as np

from scripts.fig2b_hf_scan import DEFAULT_ALPHA_VALUES
from src import lf_mp, lf_mr, my_direct_ep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ALPHA = 2.4
DEFAULT_ALPHA_VALUES_PREBREAK = DEFAULT_ALPHA_VALUES[DEFAULT_ALPHA_VALUES < SOURCE_ALPHA]
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "fig2b_mr_broken_seed.csv"
DEFAULT_FRAMES_PATH = PROJECT_ROOT / "data" / "fig2b_mr_broken_seed_frames.npz"
FRAME_RECIPE = "target_best_plus_cs_orbits_plus_source_best_orbit"
CSV_COLUMNS = (
    "alpha", "g", "omega", "L", "t", "coupling_convention",
    "source_alpha", "source_lf_energy", "source_density_imbalance",
    "nrandom", "random_scale", "seed", "gtol", "max_cycle",
    "frame_recipe", "overlap_cut", "n_frames_base", "n_frames_augmented",
    "rank_base", "rank_augmented", "energy_base", "energy_augmented",
    "energy_gain", "n0", "n1", "n2", "n3", "density_imbalance",
    "norm_error", "min_retained_overlap",
    "energy_cut_loose", "rank_cut_loose",
    "energy_cut_tight", "rank_cut_tight",
)


def _solve(tmat, g, omega, lam, shift, overlap_cut):
    smat = lf_mr.lf_frame_overlap(lam, shift)
    hmat = lf_mr.lf_frame_hamiltonian(tmat, g, omega, lam, shift)
    energy, coeff, rank = lf_mr.lf_noci_lowest(
        hmat, smat, overlap_cut=overlap_cut
    )
    return float(energy), coeff, int(rank), smat, hmat


def run_broken_seed_scan(
    alpha_values: np.ndarray,
    *,
    source_alpha: float = SOURCE_ALPHA,
    nrandom: int = 8,
    random_scale: float = 0.5,
    seed: int = 0,
    gtol: float = 1e-8,
    max_cycle: int = 500,
    overlap_cut: float = 1e-10,
    csv_path: str | Path | None = None,
    frames_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Reoptimize target LF-HF frames, then add one fixed broken-frame orbit.

    A frame has lam[x,p] and shift[x] with float64 shapes (L,L) and (L,).
    The source is optimized once at source_alpha. Each target uses its own
    g=sqrt(alpha*omega) and the paper's uncentered Hamiltonian; only the
    source frame parameters are transferred. The target best/CS and source
    full translation orbits have shape (L,L,L), ordered [R,x,p].
    """
    alpha_values = np.asarray(alpha_values)
    if (alpha_values.ndim != 1 or alpha_values.size == 0
            or np.iscomplexobj(alpha_values)
            or not np.all(np.isfinite(alpha_values))
            or np.any(alpha_values < 0)):
        raise ValueError("alpha_values must be a nonempty real nonnegative 1D array")
    if (not np.isfinite(source_alpha) or source_alpha <= 0
            or np.any(alpha_values >= source_alpha)):
        raise ValueError("all target alpha values must be below source_alpha")
    if not np.isfinite(overlap_cut) or overlap_cut <= 0:
        raise ValueError("overlap_cut must be finite and positive")

    omega = 0.5
    length = 4
    tmat = my_direct_ep.electron_ring_hopping(length, -1.0)
    settings = dict(
        nrandom=nrandom, random_scale=random_scale, seed=seed,
        gtol=gtol, max_cycle=max_cycle,
    )
    source, _ = lf_mp.lf_hf_full_alpha_point(
        source_alpha, tmat, omega, **settings
    )
    if source.density_imbalance < 1e-3:
        raise RuntimeError("source LF-HF frame is not symmetry broken")
    source_lam, source_shift = lf_mr.lf_translation_orbit(
        source.lam, source.shift
    )

    rows = []
    frame_best = []
    frame_best_shift = []
    frame_cs = []
    frame_cs_shift = []
    csv_file = None
    try:
        if csv_path is not None:
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = csv_path.open("w", newline="", encoding="utf-8")
            writer = csv.DictWriter(
                csv_file, fieldnames=CSV_COLUMNS, lineterminator="\n"
            )
            writer.writeheader()
            csv_file.flush()

        for alpha_value in alpha_values:
            alpha = float(alpha_value)
            g = float(my_direct_ep.alpha_to_g(alpha, omega))
            best, results = lf_mp.lf_hf_full_alpha_point(
                alpha, tmat, omega, **settings
            )
            cs_start = results[0]
            if not cs_start.success:
                raise RuntimeError(f"CS-start LF-HF branch failed at alpha={alpha}")
            if best.density_imbalance > 1e-5:
                raise RuntimeError(f"target LF-HF branch is broken at alpha={alpha}")

            best_lam, best_shift = lf_mr.lf_translation_orbit(
                best.lam, best.shift
            )
            cs_lam, cs_shift = lf_mr.lf_translation_orbit(
                cs_start.lam, cs_start.shift
            )
            base_lam = np.concatenate((best_lam, cs_lam))
            base_shift = np.concatenate((best_shift, cs_shift))
            all_lam = np.concatenate((base_lam, source_lam))
            all_shift = np.concatenate((base_shift, source_shift))
            base_energy, _, base_rank, _, _ = _solve(
                tmat, g, omega, base_lam, base_shift, overlap_cut
            )
            energy, coeff, rank, smat, hmat = _solve(
                tmat, g, omega, all_lam, all_shift, overlap_cut
            )
            if energy > base_energy + 1e-8:
                raise RuntimeError(f"adding frames raised the energy at alpha={alpha}")
            density = lf_mr.lf_noci_site_density(coeff, smat)
            norm_error = abs(float(np.sum(density)) - 1.0)
            if norm_error > 1e-9:
                raise RuntimeError(f"NOCI state is not normalized at alpha={alpha}")
            eigenvalues = np.linalg.eigvalsh(
                smat.reshape(coeff.size, coeff.size)
            )
            retained = eigenvalues[
                eigenvalues > overlap_cut * max(1.0, float(eigenvalues[-1]))
            ]
            if len(retained) != rank:
                raise RuntimeError("overlap rank is inconsistent")
            loose_energy, _, loose_rank = lf_mr.lf_noci_lowest(
                hmat, smat, overlap_cut=overlap_cut * 100
            )
            tight_energy, _, tight_rank = lf_mr.lf_noci_lowest(
                hmat, smat, overlap_cut=overlap_cut / 100
            )
            row = dict(
                alpha=alpha, g=g, omega=omega, L=length, t=-1.0,
                coupling_convention="paper_uncentered",
                source_alpha=float(source_alpha),
                source_lf_energy=float(source.fun),
                source_density_imbalance=float(source.density_imbalance),
                nrandom=nrandom, random_scale=random_scale, seed=seed,
                gtol=gtol, max_cycle=max_cycle, frame_recipe=FRAME_RECIPE,
                overlap_cut=overlap_cut,
                n_frames_base=len(base_lam), n_frames_augmented=len(all_lam),
                rank_base=base_rank, rank_augmented=rank,
                energy_base=base_energy, energy_augmented=energy,
                energy_gain=base_energy - energy,
                n0=float(density[0]), n1=float(density[1]),
                n2=float(density[2]), n3=float(density[3]),
                density_imbalance=float(np.ptp(density)),
                norm_error=norm_error,
                min_retained_overlap=float(retained[0]),
                energy_cut_loose=float(loose_energy),
                rank_cut_loose=int(loose_rank),
                energy_cut_tight=float(tight_energy),
                rank_cut_tight=int(tight_rank),
            )
            rows.append(row)
            frame_best.append(best.lam.copy())
            frame_best_shift.append(best.shift.copy())
            frame_cs.append(cs_start.lam.copy())
            frame_cs_shift.append(cs_start.shift.copy())
            print(
                f"alpha={alpha:.1f} base={base_energy:.9f} "
                f"augmented={energy:.9f} rank={rank}", flush=True
            )
            if csv_file is not None:
                writer.writerow({
                    key: f"{value:.16g}" if isinstance(value, float) else value
                    for key, value in row.items()
                })
                csv_file.flush()
    finally:
        if csv_file is not None:
            csv_file.close()

    if frames_path is not None:
        frames_path = Path(frames_path)
        frames_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            frames_path,
            source_alpha=np.float64(source_alpha),
            source_lam=source.lam,
            source_shift=source.shift,
            target_alpha=np.asarray(alpha_values, dtype=np.float64),
            target_best_lam=np.stack(frame_best),
            target_best_shift=np.stack(frame_best_shift),
            target_cs_lam=np.stack(frame_cs),
            target_cs_shift=np.stack(frame_cs_shift),
        )
    return rows


if __name__ == "__main__":
    completed = run_broken_seed_scan(
        DEFAULT_ALPHA_VALUES_PREBREAK,
        csv_path=DEFAULT_CSV_PATH,
        frames_path=DEFAULT_FRAMES_PATH,
    )
    print(f"Saved {len(completed)} points to {DEFAULT_CSV_PATH}")
    print(f"Saved frame parameters to {DEFAULT_FRAMES_PATH}")
