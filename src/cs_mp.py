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


def local_ep_coupling_mo(
    coupling: float,
    mo_coeff: np.ndarray,
) -> np.ndarray:
    """Transform local Holstein couplings from the site basis to the MO basis.

    Parameters
    ----------
    coupling
        Real local electron--phonon coupling strength ``g``.
    mo_coeff
        Canonical MO coefficients with shape ``(nsite, nmo)`` and element
        ``mo_coeff[x, p] = C[x, p]``.  The current one-mode-per-site model
        requires a square matrix, so ``nsite == nmo``.

    Returns
    -------
    numpy.ndarray
        Coupling tensor with shape ``(nmode, nmo, nmo)`` and index order
        ``coupling_mo[x, p, q] = g * C[x, p].conj() * C[x, q]``.  Complex
        MO coefficients produce a complex tensor.
    """
    if mo_coeff.ndim != 2 or mo_coeff.shape[1] != mo_coeff.shape[0]:
        raise ValueError("mo_coeff must be a square matrix")
    norb = mo_coeff.shape[0]
    G = np.zeros((norb,norb,norb), dtype = np.result_type(coupling, mo_coeff))
    for x in range(norb):
        for p in range(norb):
            for q in range(norb):
                G[x,p,q] = coupling * mo_coeff[x,p].conj() * mo_coeff[x,q]
    return G


def cs_mp2_mixed_intermediates(
    mo_energy: np.ndarray,
    coupling_mo: np.ndarray,
    omega: float,
    nocc: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build mixed electronic-single plus one-phonon CS-MP2 intermediates.

    Parameters
    ----------
    mo_energy
        Canonical MO energies with shape ``(nmo,)``.  Occupied orbitals must
        precede virtual orbitals.
    coupling_mo
        MO-basis electron--phonon couplings with shape
        ``(nmode, nmo, nmo)`` and index order ``[x, p, q]``.
    omega
        Phonon frequency used in every mode of the Fig. 2b model.
    nocc
        Number of occupied MOs, satisfying ``1 <= nocc < nmo``.

    Returns
    -------
    matrix_elements
        Excitation matrix elements ``G[x, a, i]`` arranged with shape
        ``(nmode, nocc, nvir)`` and index order ``[x, i, a]``.  The virtual
        subspace index ``a`` corresponds to global MO index ``nocc + a``.
    denominators
        Real zeroth-order denominators ``epsilon_i - epsilon_a - omega``
        with the same shape and index order as ``matrix_elements``.
    """
    if mo_energy.ndim != 1:
        raise ValueError("mo_energy must be a 1D array")
    nmo = mo_energy.shape[0]
    if coupling_mo.ndim != 3 or coupling_mo.shape[1] != nmo or coupling_mo.shape[2] != nmo:
        raise ValueError("coupling_mo must be a 3D array with shape (nmode, nmo, nmo)")
    if nocc < 1 or nocc >= nmo:
        raise ValueError("nocc must be between 1 and nmo-1")
    matrix_elements = coupling_mo[:, nocc:, :nocc]  # shape (nmode, nvir, nocc)
    matrix_elements = matrix_elements.transpose(0, 2, 1)  # shape (nmode, nocc, nvir)
    denominators = (mo_energy[:nocc, None] - mo_energy[None, nocc:] - omega)
    denominators = np.broadcast_to(denominators, (coupling_mo.shape[0], nocc, nmo - nocc))  # shape (nmode, nocc, nvir)
    return matrix_elements, denominators


def cs_mp2_mixed_energy(
    matrix_elements: np.ndarray,
    denominators: np.ndarray,
) -> float:
    """Return the mixed electron--phonon contribution to the CS-MP2 energy.

    Both inputs use shape ``(nmode, nocc, nvir)`` and index order
    ``[x, i, a]``.  The correction is
    ``sum_xia(abs(matrix_elements[x, i, a])**2 / denominators[x, i, a])``.
    Denominators must be real and strictly negative.

    Returns
    -------
    float
        The one-electron CS-MP2 second-order correction.  For the target
        system this is the complete second-order correction because pure
        phonon excitations vanish at the CS stationary point and electronic
        double excitations do not exist.
    """
    if matrix_elements.ndim != 3 or denominators.ndim != 3:
        raise ValueError("matrix_elements and denominators must be 3D arrays")
    if matrix_elements.shape != denominators.shape:
        raise ValueError("matrix_elements and denominators must have the same shape")
    for denominator in denominators.flat:
        if denominator >= 0 or not np.isreal(denominator):
            raise ValueError("Denominator must be negative for stability")
    energy_contributions = (np.abs(matrix_elements)**2) / denominators
    return float(np.sum(energy_contributions))


def cs_mp2_point(
    tmat: np.ndarray,
    coupling: float,
    omega: float,
    coeff0: np.ndarray,
    *,
    conv_tol: float = 1e-10,
    max_cycle: int = 100,
) -> dict[str, object]:
    """Solve one complete one-electron CS-HF/CS-MP2 parameter point.

    The CS-HF reference is converged first.  Its final Fock matrix is then
    canonicalized, the local coupling is transformed to the MO basis, and
    the mixed electron--phonon second-order correction is evaluated for one
    occupied MO.

    Parameters
    ----------
    tmat
        Hermitian site-basis hopping matrix with shape ``(L, L)``.
    coupling
        Real local electron--phonon coupling strength ``g``.
    omega
        Positive phonon frequency.
    coeff0
        Initial site-basis occupied orbital with shape ``(L,)``.
    conv_tol
        CS-HF density convergence threshold.
    max_cycle
        Maximum number of CS-HF iterations.

    Returns
    -------
    dict
        Auditable result containing ``hf_energy``, ``mp2_correction``,
        ``total_energy``, ``coeff``, ``shift``, ``mo_energy``, ``mo_coeff``,
        ``niter``, and ``density_residual``.  The total energy equals the
        CS-HF total energy plus the second-order correction.

    Raises
    ------
    RuntimeError
        If the CS-HF reference does not converge.
    """
    energy, coeff, shift, converged, niter, density_residual = cs_hf_scf(tmat, coupling, omega, coeff0, conv_tol=conv_tol, max_cycle=max_cycle)
    if not converged:
        raise RuntimeError("CS-HF SCF did not converge")
    F = cs_fock(tmat, coupling, shift)
    mo_energy, mo_coeff = np.linalg.eigh(F)
    local_coupling_mo = local_ep_coupling_mo(coupling, mo_coeff)
    nocc = 1
    matrix_elements, denominators = cs_mp2_mixed_intermediates(mo_energy, local_coupling_mo, omega, nocc)
    mp2_energy = cs_mp2_mixed_energy(matrix_elements, denominators)
    Energy_total = energy + mp2_energy
    return {
        "hf_energy": energy,
        "mp2_correction": mp2_energy,
        "total_energy": Energy_total,
        "coeff": coeff,
        "shift": shift,
        "mo_energy": mo_energy,
        "mo_coeff": mo_coeff,
        "niter": niter,
        "density_residual": density_residual,
    }
