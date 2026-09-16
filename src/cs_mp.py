"""Coherent-state Hartree--Fock building blocks for Fig. 2b.

This stage provides the one-electron CS-HF energy, stationary displacement,
Fock matrix, and self-consistent-field driver.  CS-MP corrections are deferred
to a later stage.
"""

import numpy as np


def cs_site_density(coeff: np.ndarray) -> np.ndarray:
    """Return ``n[x] = |coeff[x]|**2`` for one normalized occupied orbital.

    Parameters
    ----------
    coeff
        Site-basis occupied-orbital coefficients with shape ``(L,)``.  Real
        and complex coefficients must both be supported.
    """
    return np.abs(coeff)**2


def cs_stationary_shift(
    coeff: np.ndarray,
    g: float,
    omega: float,
) -> np.ndarray:
    """Return the stationary coherent displacement ``z[x] = -g*n[x]/omega``."""
    return -g * cs_site_density(coeff) / omega


def cs_fock(tmat: np.ndarray, g: float, shift: np.ndarray) -> np.ndarray:
    """Build the one-electron CS-HF Fock matrix ``t + diag(2*g*shift)``."""
    return tmat + np.diag(2 * g * shift)


def cs_energy(
    tmat: np.ndarray,
    g: float,
    omega: float,
    coeff: np.ndarray,
    shift: np.ndarray,
) -> float:
    """Evaluate ``c† t c + omega*z·z + 2*g*z·n``.

    This function receives ``shift`` explicitly so that its three energy
    contributions can be checked independently.  It must return a real Python
    ``float`` for a Hermitian ``tmat``.
    """
    return np.vdot(coeff, tmat @ coeff).real + omega * np.dot(shift, shift) + 2 * g * np.dot(shift, cs_site_density(coeff))


def cs_hf_scf(
    tmat: np.ndarray,
    g: float,
    omega: float,
    coeff0: np.ndarray,
    *,
    conv_tol: float = 1e-10,
    max_cycle: int = 100,
) -> tuple[float, np.ndarray, np.ndarray, bool, int, float]:
    """Solve the one-electron CS-HF fixed-point equations.

    Parameters
    ----------
    tmat
        Hermitian site-basis hopping matrix with shape ``(L, L)``.
    g
        Electron--phonon coupling strength.
    omega
        Phonon frequency entering ``z[x] = -g*n[x]/omega``.
    coeff0
        Initial occupied-orbital coefficients with shape ``(L,)``.  The
        driver normalizes a copy, so the caller's array is not modified.
    conv_tol
        Convergence threshold for the maximum site-density change.
    max_cycle
        Maximum number of undamped fixed-point iterations.

    Returns
    -------
    energy
        CS-HF total energy, not the occupied Fock eigenvalue.
    coeff
        Final normalized site-basis occupied orbital, shape ``(L,)``.
    shift
        Stationary displacement belonging to ``coeff``, shape ``(L,)``.
    converged
        Whether the density residual fell below ``conv_tol``.
    niter
        Number of Fock diagonalizations performed.
    density_residual
        Maximum site-density change in the last iteration.
    """
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("tmat must be a square matrix")
    if coeff0.ndim != 1 or coeff0.shape[0] != tmat.shape[0]:
        raise ValueError("coeff0 must be a vector of the same length as tmat")
    coeff = coeff0.copy()
    # normalize the initial guess
    norm = np.linalg.norm(coeff)
    if norm == 0:
        raise ValueError("Initial guess coeff0 cannot be the zero vector")
    coeff /= norm
    converged = False
    niter = 0
    density_residual = float("inf")
    for cycle in range(max_cycle):
        old_density = cs_site_density(coeff)
        old_shift = cs_stationary_shift(coeff, g, omega)
        fock = cs_fock(tmat, g, old_shift)
        new_coeff = np.linalg.eigh(fock)[1][:, 0]  # lowest eigenvector
        new_density = cs_site_density(new_coeff)

        density_residual = float(np.max(np.abs(new_density - old_density)))

        coeff = new_coeff
        niter += 1

        if density_residual < conv_tol:
            converged = True
            break

    shift = cs_stationary_shift(coeff, g, omega)
    energy = float(cs_energy(tmat, g, omega, coeff, shift))
    return energy, coeff, shift, converged, niter, density_residual
