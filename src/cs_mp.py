"""Coherent-state mean-field and perturbation theory for Fig. 2b.

Stage 1 intentionally contains only the implemented CS-HF primitives.  The SCF
driver and CS-MP2 are deferred until the theory stage has been accepted.
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
