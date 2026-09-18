"""Lang--Firsov mean-field reference and perturbative corrections."""

import numpy as np

from .cs_mp import cs_site_density


def lf_franck_condon(lam: np.ndarray) -> np.ndarray:
    """Return the zero-phonon Franck--Condon overlap matrix.

    Parameters
    ----------
    lam
        Real conditional displacements with shape ``(nmode, norb)`` and
        index order ``lam[x, p]``.

    Returns
    -------
    numpy.ndarray
        Matrix with shape ``(norb, norb)`` whose elements are
        ``exp(-0.5 * sum_x((lam[x, p] - lam[x, q])**2))``.
    """
    if lam.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")
    norb = lam.shape[1]
    S = np.zeros((norb, norb), dtype=np.float64)
    for p in range(norb):
        for q in range(norb):
            delta = lam[:, p] - lam[:, q]
            S[p, q] = np.exp(-0.5 * np.sum(delta**2))
    return S


def lf_stationary_shift(
    coeff: np.ndarray,
    lam: np.ndarray,
    g: float,
    omega: float,
) -> np.ndarray:
    """Return the stationary coherent displacement for one electron.

    Parameters
    ----------
    coeff
        Normalized site-basis orbital with shape ``(norb,)``.  Real and
        complex coefficients are both supported.
    lam
        Real conditional displacements with shape ``(norb, norb)`` and
        index order ``lam[x, p]``.  This local Holstein specialization
        requires one phonon mode per electronic site.
    g
        Local electron--phonon coupling strength.
    omega
        Positive phonon frequency.

    Returns
    -------
    numpy.ndarray
        Real displacement ``lam @ density - (g / omega) * density`` with
        shape ``(norb,)``.
    """
    if coeff.ndim != 1:
        raise ValueError("Coefficient array must be 1-dimensional.")
    if lam.ndim != 2:
        raise ValueError("Displacement array must be 2-dimensional.")
    if lam.shape != (coeff.size, coeff.size):
        raise ValueError("Displacement array dimensions must match coefficient array length.")
    if omega <= 0:
        raise ValueError("Frequency omega must be positive.")
    density = cs_site_density(coeff)
    conditional_shift = np.dot(lam, density)
    shift = conditional_shift - (g / omega) * density
    return shift


def lf_effective_one_body(
    tmat: np.ndarray,
    g: float,
    omega: float,
    shift: np.ndarray,
    lam: np.ndarray,
) -> np.ndarray:
    """Build the one-electron zero-phonon LF effective Hamiltonian.

    Parameters
    ----------
    tmat
        Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Local electron--phonon coupling strength.
    omega
        Positive phonon frequency.
    shift
        Real coherent displacement with shape ``(norb,)``.
    lam
        Real conditional displacements with shape ``(norb, norb)`` and
        index order ``lam[x, p]``.

    Returns
    -------
    numpy.ndarray
        Effective one-body matrix ``tmat * overlap + diag(correction)`` with
        shape ``(norb, norb)``.  The global constant ``omega * shift @ shift``
        is not included.
    """
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("Hopping matrix must be 2-dimensional and square.")
    if lam.shape != tmat.shape:
        raise ValueError("Displacement array dimensions must match hopping matrix dimensions.")
    if shift.ndim != 1 or shift.size != tmat.shape[0]:
        raise ValueError("Shift array must be 1-dimensional and match hopping matrix size.")
    if omega <= 0:
        raise ValueError("Frequency omega must be positive.")
    norb = tmat.shape[0]
    overlap = lf_franck_condon(lam)
    diagonal_correction = np.ndarray(norb)
    for p in range(norb):
        phonon_part = omega * np.sum(lam[:, p]**2 - 2 * shift * lam[:, p])
        coupling_part = 2 * g * (shift[p] - lam[p, p])
        diagonal_correction[p] = phonon_part + coupling_part
    heff = tmat * overlap + np.diag(diagonal_correction)
    return heff


def lf_energy(
    tmat: np.ndarray,
    g: float,
    omega: float,
    coeff: np.ndarray,
    shift: np.ndarray,
    lam: np.ndarray,
) -> float:
    """Evaluate the one-electron zero-phonon LF total energy.

    Parameters
    ----------
    tmat
        Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Local electron--phonon coupling strength.
    omega
        Positive phonon frequency.
    coeff
        Normalized site-basis orbital with shape ``(norb,)``.  Real and
        complex coefficients are both supported.
    shift
        Real coherent displacement with shape ``(norb,)``.
    lam
        Real conditional displacements with shape ``(norb, norb)`` and
        index order ``lam[x, p]``.

    Returns
    -------
    float
        Total energy ``omega * shift @ shift + coeff.conj() @ heff @ coeff``.
    """
    if coeff.ndim != 1:
        raise ValueError("Coefficient array must be 1-dimensional.")
    heff = lf_effective_one_body(tmat, g, omega, shift, lam)
    if coeff.size != heff.shape[0]:
        raise ValueError("Coefficient array length must match effective Hamiltonian size.")
    electronic_energy = np.real(np.vdot(coeff, heff @ coeff))
    boson_constant = omega * np.dot(shift, shift)
    total_energy = electronic_energy + boson_constant
    return float(total_energy)
