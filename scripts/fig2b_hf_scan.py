"""Scan the CS-HF and local LF-HF branches needed for Fig. 2b."""

import numpy as np

from src import cs_mp, lf_mp, my_direct_ep


def run_hf_scan(
    alpha_values: np.ndarray,
    *,
    nrandom: int = 8,
    random_scale: float = 0.5,
    seed: int = 0,
    gtol: float = 1e-8,
    max_cycle: int = 500,
) -> list[dict[str, object]]:
    """Evaluate CS-HF and local LF-HF in the supplied coupling order.

    This experiment-level driver fixes the Fig. 2b model parameters to a
    four-site ring with ``t=-1`` and ``omega=0.5``.  CS-HF is initialized on
    its uniform branch, while every LF-HF point uses the same reproducible
    set of CS-centered random perturbations.

    Parameters
    ----------
    alpha_values
        Nonempty one-dimensional array of finite nonnegative couplings.
        Values are neither sorted nor deduplicated.
    nrandom
        Number of random LF-HF starts in addition to the CS point.
    random_scale
        Standard deviation of the LF-HF starting-point perturbations.
    seed
        Seed reused at every coupling point.
    gtol
        Convergence tolerance passed to the CS-HF and LF-HF drivers.
    max_cycle
        Maximum iteration count passed to both drivers.

    Returns
    -------
    list of dict
        One record per input coupling, in input order, containing the CS-HF
        and lowest successful LF-HF energies, densities, residuals, and
        convergence metadata.

    Raises
    ------
    RuntimeError
        If the symmetric CS-HF calculation fails to converge at any point.
    """
    norb = 4
    omega = 0.5
    t = -1.0
    tmat = my_direct_ep.electron_ring_hopping(norb, t)
    cs_coeff0 = np.full(norb, 1.0 / np.sqrt(norb))
    records = []
    if alpha_values.ndim != 1 or alpha_values.size == 0:
        raise ValueError("alpha_values must be a non-empty 1D array")
    if (
        np.iscomplexobj(alpha_values)
        or not np.all(np.isfinite(alpha_values))
        or np.any(alpha_values < 0)
    ):
        raise ValueError("alpha_values must be a real, finite, non-negative array")
    for alpha in alpha_values:
        alpha = float(alpha)
        g = float(my_direct_ep.alpha_to_g(alpha, omega))
        (
            cs_energy,
            cs_coeff,
            _,
            cs_converged,
            cs_niter,
            cs_density_residual,
        ) = cs_mp.cs_hf_scf(
            tmat,
            g,
            omega,
            cs_coeff0,
            conv_tol=gtol,
            max_cycle=max_cycle,
        )
        if not cs_converged:
            raise RuntimeError(f"CS-HF failed to converge for alpha={alpha}, g={g}")
        cs_density = cs_mp.cs_site_density(cs_coeff)
        cs_density_imbalance = float(np.max(cs_density) - np.min(cs_density))
        best, _ = lf_mp.lf_hf_local_alpha_point(
            alpha,
            tmat,
            omega,
            nrandom=nrandom,
            random_scale=random_scale,
            seed=seed,
            gtol=gtol,
            max_cycle=max_cycle,
        )
        record = {
            "alpha": alpha,
            "g": g,
            "cs_energy": float(cs_energy),
            "cs_density": cs_density.copy(),
            "cs_density_imbalance": cs_density_imbalance,
            "cs_niter": cs_niter,
            "cs_density_residual": cs_density_residual,
            "lf_energy": float(best.fun),
            "lf_density": best["density"].copy(),
            "lf_density_imbalance": best["density_imbalance"],
            "lf_nstart": best["nstart"],
            "lf_nconverged": best["nconverged"],
            "lf_shift_residual": best["shift_residual"],
        }
        records.append(record)
    return records
