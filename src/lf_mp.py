"""Lang--Firsov mean-field reference and perturbative corrections."""

import math
from itertools import combinations

import numpy as np
from scipy import optimize as scipy_optimize
from scipy import linalg as scipy_linalg

from .cs_mp import cs_site_density


def lf_total_phonon_configurations(nmode: int, max_total: int) -> np.ndarray:
    """Enumerate non-vacuum configurations under a total-phonon cutoff.

    Parameters
    ----------
    nmode
        Positive number of phonon modes.
    max_total
        Positive inclusive cutoff on ``sum_x occupations[k, x]``.

    Returns
    -------
    numpy.ndarray
        Nonnegative integer occupations with shape ``(nconfig, nmode)``.
        The vacuum is excluded.  Rows are ordered first by increasing total
        occupation and then lexicographically within each total.  Every row
        satisfies ``1 <= occupations[k].sum() <= max_total``.

    Notes
    -----
    This is a collective total-phonon cutoff.  It differs from the exact-ED
    tensor-product convention that independently permits ``0..Nmax`` on
    every mode.  The number of returned rows is
    ``comb(max_total + nmode, nmode) - 1``.
    """
    if not isinstance(nmode, (int, np.integer)) or isinstance(nmode, (bool, np.bool_)):
        raise TypeError("Number of modes must be a integer.")
    if nmode <= 0:
        raise ValueError("Number of modes must be positive.")
    if not isinstance(max_total, (int, np.integer)) or isinstance(max_total, (bool, np.bool_)):
        raise TypeError("Maximum total occupation must be a integer.")
    if max_total <= 0:
        raise ValueError("Maximum total occupation must be positive.")
    occupations = []
    for total in range(1, max_total + 1):
        for config in np.ndindex(*(total + 1,) * nmode):
            if sum(config) == total:
                occupations.append(config)
    return np.array(occupations, dtype=np.int64)


def lf_mp2_denominators(
    mo_energy: np.ndarray,
    omega: float,
    occupations: np.ndarray,
    nocc: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pure-phonon and electronic-single LF-MP2 denominators.

    Canonical MOs are ordered with occupied orbitals before virtual orbitals.
    For a non-vacuum phonon configuration ``X_k`` with total occupation
    ``N_k``, this function evaluates

    ``pure[k] = -N_k * omega``

    and

    ``singles[k, i, a] = epsilon[i] - epsilon[nocc+a] - N_k * omega``.

    Parameters
    ----------
    mo_energy
        Finite real canonical-MO energies in nondecreasing order with shape
        ``(nmo,)``.
    omega
        Finite positive phonon frequency shared by all modes.
    occupations
        Nonnegative integer non-vacuum configurations with shape
        ``(nconfig, nmode)`` and index order ``[k, x]``.
    nocc
        Number of occupied MOs, satisfying ``1 <= nocc < nmo``.

    Returns
    -------
    total_occupations
        Total phonon occupations ``N_k`` with shape ``(nconfig,)`` and
        ``int64`` dtype.
    pure_denominators
        Pure-phonon denominators with shape ``(nconfig,)``.
    single_denominators
        Electronic-single plus phonon denominators with shape
        ``(nconfig, nocc, nvir)`` and index order ``[k, i, a]``.  The local
        virtual index corresponds to global MO index ``nocc + a``.

    Notes
    -----
    Every returned denominator is strictly negative for a stable canonical
    reference and a non-vacuum phonon configuration.
    """
    if isinstance(nocc, (bool, np.bool_)) or not isinstance(nocc, (int, np.integer)):
        raise TypeError("Number of occupied MOs must be a positive integer.")
    if mo_energy.ndim != 1:
        raise ValueError("MO energy array must be 1-dimensional.")
    for i in range(mo_energy.size):
        if not np.isfinite(mo_energy[i]) or not np.isreal(mo_energy[i]):
            raise ValueError("MO energy array must contain finite real values.")
        if i < mo_energy.size - 1 and mo_energy[i] > mo_energy[i + 1]:
            raise ValueError("MO energy array must be nondecreasing.")
    if not np.isfinite(omega) or not np.isreal(omega) or omega <= 0:
        raise ValueError("Phonon frequency omega must be finite, real, and positive.")
    if occupations.ndim != 2:
        raise ValueError("Occupation array must be 2-dimensional.")
    for occ_value in occupations.flat:
        if not np.issubdtype(type(occ_value), np.integer) or occ_value < 0:
            raise ValueError("Occupation array must contain non-negative integers.")
    for total in np.sum(occupations, axis=1):
        if total <= 0:
            raise ValueError("Occupation array must not contain vacuum configurations.")
    nmo = mo_energy.size
    if not (1 <= nocc < nmo):
        raise ValueError("Number of occupied MOs must satisfy 1 <= nocc < nmo.")
    total_occupations = np.sum(occupations, axis=1, dtype=np.int64)
    denominator_pure = -total_occupations * omega
    G = mo_energy[:nocc, None] - mo_energy[nocc:]  # shape (nocc, nvir)
    denominator_singles = G[None, :, :] + denominator_pure[:, None, None]
    return total_occupations, denominator_pure, denominator_singles


def lf_mp2_energy(
    pure_matrix_elements: np.ndarray,
    single_matrix_elements: np.ndarray,
    pure_denominators: np.ndarray,
    single_denominators: np.ndarray,
) -> float:
    """Return the one-electron LF-MP2 second-order correction.

    For each non-vacuum phonon configuration ``X_k``, the pure-phonon
    channel has matrix element ``M[k]`` and denominator ``D[k]``.  The
    electronic-single plus phonon channel has matrix elements ``M[k, i, a]``
    and denominators ``D[k, i, a]``.  This function evaluates

    ``sum_k abs(M_pure[k])**2 / D_pure[k]``

    plus

    ``sum_kia abs(M_single[k, i, a])**2 / D_single[k, i, a]``.

    Parameters
    ----------
    pure_matrix_elements
        Finite real or complex pure-phonon amplitudes with shape
        ``(nconfig,)`` and index order ``[k]``.
    single_matrix_elements
        Finite real or complex electronic-single plus phonon amplitudes with
        shape ``(nconfig, nocc, nvir)`` and index order ``[k, i, a]``.
    pure_denominators
        Finite real strictly negative pure-phonon denominators with shape
        ``(nconfig,)``.
    single_denominators
        Finite real strictly negative electronic-single plus phonon
        denominators with shape ``(nconfig, nocc, nvir)``.

    Returns
    -------
    float
        Total LF-MP2 correction.  It is nonpositive and contains no extra
        spin, symmetry, or factorial prefactor.

    Notes
    -----
    The first axis must describe the same ordered phonon configurations in
    all four inputs.  This is the complete second-order correction for the
    one-electron Fig. 2b model; electronic double excitations do not exist.
    """
    if pure_matrix_elements.ndim != 1:
        raise ValueError("Pure-phonon matrix element array must be 1-dimensional.")
    if single_matrix_elements.ndim != 3:
        raise ValueError("Electronic-single matrix element array must be 3-dimensional.")
    if pure_denominators.ndim != 1:
        raise ValueError("Pure-phonon denominator array must be 1-dimensional.")
    if single_denominators.ndim != 3:
        raise ValueError("Electronic-single denominator array must be 3-dimensional.")
    if pure_matrix_elements.shape[0] != pure_denominators.shape[0]:
        raise ValueError("Pure-phonon matrix elements and denominators must have the same length.")
    if single_matrix_elements.shape != single_denominators.shape:
        raise ValueError("Electronic-single matrix elements and denominators must have the same shape.")
    if pure_matrix_elements.shape[0] != single_matrix_elements.shape[0]:
        raise ValueError("Pure-phonon and electronic-single matrix elements must have the same first dimension.")
    if not np.all(np.isfinite(pure_denominators)) or not np.all(np.isreal(pure_denominators)) or not np.all(pure_denominators < 0):
        raise ValueError("Pure-phonon denominators must be finite, real, and strictly negative.")
    if not np.all(np.isfinite(single_denominators)) or not np.all(np.isreal(single_denominators)) or not np.all(single_denominators < 0):
        raise ValueError("Electronic-single denominators must be finite, real, and strictly negative.")
    if not np.all(np.isfinite(pure_matrix_elements)):
        raise ValueError("Pure-phonon matrix elements must be finite.")
    if not np.all(np.isfinite(single_matrix_elements)):
        raise ValueError("Electronic-single matrix elements must be finite.")
    energy_pure = np.sum(np.abs(pure_matrix_elements)**2 / pure_denominators)
    energy_single = np.sum(np.abs(single_matrix_elements)**2 / single_denominators)
    total_energy = energy_pure + energy_single
    return float(total_energy)


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


def lf_displacement_vacuum_amplitudes(
    lam: np.ndarray,
    occupations: np.ndarray,
) -> np.ndarray:
    """Evaluate vacuum-to-number-state LF displacement amplitudes.

    For the hopping operator ``a_p^dagger a_q``, define the displacement
    vector ``delta[x, p, q] = lam[x, p] - lam[x, q]``.  This function
    evaluates

    ``<X| exp(sum_x delta[x,p,q] * (b_x^dagger-b_x)) |0>``

    for every supplied multimode occupation vector ``X``.

    Parameters
    ----------
    lam
        Finite real LF parameters with shape ``(nmode, norb)`` and index
        order ``lam[x, p]``.
    occupations
        Nonnegative integer phonon occupations with shape
        ``(nconfig, nmode)`` and index order ``occupations[k, x]``.

    Returns
    -------
    numpy.ndarray
        Real amplitudes with shape ``(nconfig, norb, norb)`` and index
        order ``[k, p, q]``.  Configuration ``k`` represents
        ``|X_k> = |n_0, n_1, ...>``.  Odd total phonon occupations retain
        the sign of the displacement components.
    """
    if lam.ndim != 2:
        raise ValueError("LF parameter array must be 2-dimensional.")
    for lam_value in lam.flat:
        if not np.isfinite(lam_value) or not np.isreal(lam_value):
            raise ValueError("LF parameter array must contain finite real values.")
    if occupations.ndim != 2:
        raise ValueError("Occupation array must be 2-dimensional.")
    nmode, norb = lam.shape
    if occupations.shape[1] != nmode:
        raise ValueError("Occupation array second dimension must match LF parameter first dimension.")
    for occ_value in occupations.flat:
        if not np.issubdtype(type(occ_value), np.integer) or occ_value < 0:
            raise ValueError("Occupation array must contain non-negative integers.")
    delta = lam[:, :, None] - lam[:, None, :]
    Gpq = np.exp(-0.5 * np.sum(delta**2, axis=0))
    amplitudes = np.empty((occupations.shape[0], norb, norb), dtype=np.float64)
    for k in range(occupations.shape[0]):
        one = np.ones((norb, norb), dtype=np.float64)
        for x in range(nmode):
            n = occupations[k, x]
            if n > 0:
                one *= delta[x]**n / np.sqrt(math.factorial(int(n)))
        amplitudes[k] = Gpq * one
    return amplitudes


def lf_vacuum_coupling_site_matrices(
    tmat: np.ndarray,
    g: float,
    omega: float,
    shift: np.ndarray,
    lam: np.ndarray,
    occupations: np.ndarray,
) -> np.ndarray:
    """Build non-vacuum LF fluctuation couplings in the site basis.

    For every non-vacuum phonon configuration ``X_k``, construct the
    one-electron matrix

    ``W[k, p, q] = <X_k, p| V |0, q>``.

    Its hopping contribution is ``tmat[p, q]`` times the corresponding
    vacuum-to-number-state displacement amplitude.  A configuration with
    exactly one phonon in mode ``x`` additionally receives the diagonal
    residual coupling

    ``omega * shift[x] + g * delta[x, p] - omega * lam[x, p]``.

    Parameters
    ----------
    tmat
        Finite Hermitian site-basis hopping matrix with shape
        ``(norb, norb)``.
    g
        Finite real local Holstein coupling strength.
    omega
        Finite positive phonon frequency shared by all modes.
    shift
        Finite real coherent displacements with shape ``(nmode,)`` and
        index order ``shift[x]``.
    lam
        Finite real LF parameters with shape ``(nmode, norb)`` and index
        order ``lam[x, p]``.  This local Holstein specialization requires
        ``nmode == norb``.
    occupations
        Nonnegative integer configurations with shape ``(nconfig, nmode)``.
        Every row must have a strictly positive total occupation.

    Returns
    -------
    numpy.ndarray
        Site-basis coupling matrices with shape ``(nconfig, norb, norb)``
        and index order ``[k, p, q]``.  The dtype preserves a complex
        ``tmat``.  A matrix for fixed ``X_k`` need not be Hermitian.

    Notes
    -----
    Vacuum configurations are rejected because the zero-phonon block also
    contains static LF terms and the zeroth-order Fock subtraction.  For
    non-vacuum configurations those terms have zero matrix element, so the
    returned matrix is directly the required fluctuation coupling.
    """
    if lam.ndim != 2 or lam.shape[0] != lam.shape[1]:
        raise ValueError("LF parameter array must be 2-dimensional and square.")
    norb = lam.shape[1]
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1] or tmat.shape[0] != norb or not np.allclose(tmat, tmat.conj().T):
        raise ValueError("Hopping matrix must be 2-dimensional, square, and Hermitian.")
    for tmat_value in tmat.flat:
        if not np.isfinite(tmat_value):
            raise ValueError("Hopping matrix must contain finite values.")
    if shift.ndim != 1 or shift.size != norb:
        raise ValueError("Shift array must be 1-dimensional and match LF parameter first dimension.")
    for shift_value in shift.flat:
        if not np.isfinite(shift_value) or not np.isreal(shift_value):
            raise ValueError("Shift array must contain finite real values.")
    if occupations.ndim != 2 or occupations.shape[1] != norb:
        raise ValueError("Occupation array must be 2-dimensional with second dimension matching LF parameter first dimension.")
    for occ_value in occupations.flat:
        if not np.issubdtype(type(occ_value), np.integer) or occ_value < 0:
            raise ValueError("Occupation array must contain non-negative integers.")
    if np.any(np.sum(occupations, axis=1) == 0):
        raise ValueError("Occupation array must not contain vacuum configurations.")
    if omega <= 0 or not np.isfinite(omega) or not np.isreal(omega):
        raise ValueError("Frequency omega must be positive.")
    if not np.isfinite(g) or not np.isreal(g):
        raise ValueError("Coupling g must be finite and real.")
    amplitudes = lf_displacement_vacuum_amplitudes(lam, occupations)
    W = np.empty((occupations.shape[0], norb, norb), dtype=np.result_type(tmat, np.float64))
    for k in range(occupations.shape[0]):
        W[k] = tmat * amplitudes[k]
        if np.sum(occupations[k]) == 1:
            x = np.argmax(occupations[k])
            residual = np.empty(norb, dtype=np.float64)
            for p in range(norb):
                residual[p] = omega * shift[x] + g * (1 if p == x else 0) - omega * lam[x, p]
            W[k] += np.diag(residual)
    return W


def lf_vacuum_coupling_mo_elements(
    site_couplings: np.ndarray,
    mo_coeff: np.ndarray,
    nocc: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform non-vacuum LF couplings and extract MP2 channels.

    Each site-basis matrix is transformed with the same orthonormal canonical
    MO coefficients according to ``W_mo[k] = C.conj().T @ W_site[k] @ C``.
    Occupied MOs must precede virtual MOs.  For the one-body operator, the
    unchanged-determinant amplitude is the occupied trace, while the
    determinant-single amplitude is ``W_mo[k, a, i]``.

    Parameters
    ----------
    site_couplings
        Non-vacuum site-basis matrices with shape
        ``(nconfig, nsite, nsite)`` and index order ``[k, p, q]``.
        A matrix for fixed configuration need not be Hermitian.
    mo_coeff
        Finite square unitary canonical-MO coefficient matrix with shape
        ``(nsite, nmo)`` and index order ``[p, m]``.  Columns are MOs and
        ``nsite == nmo`` for the complete site basis used here.
    nocc
        Number of occupied MOs, satisfying ``1 <= nocc < nmo``.  The Fig. 2b
        one-electron problem uses ``nocc = 1``.

    Returns
    -------
    mo_couplings
        All transformed matrices with shape ``(nconfig, nmo, nmo)`` and
        index order ``[k, m, n]``.
    pure_phonon
        Unchanged-determinant amplitudes
        ``sum_i W_mo[k, i, i]`` with shape ``(nconfig,)``.
    singles
        Single-excitation amplitudes ``W_mo[k, a, i]`` with shape
        ``(nconfig, nocc, nvir)`` and index order ``[k, i, a]``.  The local
        virtual index ``a`` corresponds to global MO index ``nocc + a``.

    Notes
    -----
    The returned occupied trace is the complete pure-phonon electronic
    matrix element for the one-electron Fig. 2b problem.  Additional
    many-electron LF two-body terms would need separate treatment.
    """
    if site_couplings.ndim != 3:
        raise ValueError("Site-basis coupling array must be 3-dimensional.")
    for site_value in site_couplings.flat:
        if not np.isfinite(site_value):
            raise ValueError("Site-basis coupling array must contain finite values.")
    nconfig, nsite, nsite2 = site_couplings.shape
    if nsite != nsite2:
        raise ValueError("Site-basis coupling matrices must be square.")
    if mo_coeff.ndim != 2 or mo_coeff.shape[0] != mo_coeff.shape[1] or mo_coeff.shape[0] != nsite:
        raise ValueError("MO coefficient array must be square and match site-basis size.")
    for mo_value in mo_coeff.flat:
        if not np.isfinite(mo_value):
            raise ValueError("MO coefficient array must contain finite values.")
    nmo = mo_coeff.shape[1]
    if not isinstance(nocc, (int, np.integer)) or isinstance(nocc, (bool, np.bool_)):
        raise TypeError("Number of occupied MOs must be a positive integer.")
    if not (1 <= nocc < nmo):
        raise ValueError("Number of occupied MOs must satisfy 1 <= nocc < nmo.")
    C = mo_coeff
    if not np.allclose(C.conj().T @ C, np.eye(nmo)):
        raise ValueError("MO coefficient array must be unitary.")
    W_mo = np.empty((nconfig, nmo, nmo), dtype=np.result_type(site_couplings, np.complex128))
    for k in range(nconfig):
        W_mo[k] = C.conj().T @ site_couplings[k] @ C
    pure_phonon = np.array([np.trace(W_mo[k, :nocc, :nocc]) for k in range(nconfig)], dtype=np.complex128)
    singles = W_mo[:, nocc:, :nocc].transpose(0, 2, 1)
    return W_mo, pure_phonon, singles


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
    diagonal_correction = np.zeros(norb)
    for p in range(norb):
        phonon_part = omega * np.sum(lam[:, p]**2 - 2 * shift * lam[:, p])
        coupling_part = 2 * g * (shift[p] - lam[p, p])
        diagonal_correction[p] = phonon_part + coupling_part
    heff = tmat * overlap + np.diag(diagonal_correction)
    return heff


def lf_mp2_reference_point(
    tmat: np.ndarray,
    g: float,
    omega: float,
    shift: np.ndarray,
    lam: np.ndarray,
    *,
    max_total: int,
) -> dict[str, object]:
    """Evaluate LF-HF and LF-MP2 for one fixed one-electron LF reference.

    The final zero-phonon effective Hamiltonian is diagonalized to define the
    canonical occupied and virtual MOs.  Every non-vacuum phonon configuration
    satisfying ``sum_x n_x <= max_total`` is then included in the pure-phonon
    and electronic-single LF-MP2 channels.

    Parameters
    ----------
    tmat
        Finite Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Finite real local electron--phonon coupling strength.
    omega
        Finite positive phonon frequency shared by all modes.
    shift
        Finite real coherent displacements with shape ``(norb,)``.  The
        gauge-fixed full-matrix LF-HF calculation passes a zero vector.
    lam
        Finite real conditional displacements with shape ``(norb, norb)`` and
        index order ``lam[x, p]``.
    max_total
        Positive inclusive cutoff on the total phonon occupation.

    Returns
    -------
    dict
        Auditable result with exactly the following entries:

        ``hf_energy``
            Zero-phonon LF-HF energy including ``omega * shift @ shift``.
        ``mp2_correction``
            LF-MP2 second-order correction through ``max_total``.
        ``total_energy``
            Sum of ``hf_energy`` and ``mp2_correction``.
        ``max_total``
            Integer total-phonon cutoff used for this result.
        ``coeff``
            Occupied canonical MO with shape ``(norb,)``.
        ``mo_energy``, ``mo_coeff``
            Canonical energies with shape ``(norb,)`` and site-to-MO
            coefficients with shape ``(norb, norb)``.
        ``occupations``, ``total_occupations``
            Non-vacuum configurations with shape ``(nconfig, norb)`` and
            their totals with shape ``(nconfig,)``.
        ``pure_matrix_elements``, ``pure_denominators``
            Pure-phonon channel arrays with shape ``(nconfig,)``.
        ``single_matrix_elements``, ``single_denominators``
            Electronic-single plus phonon arrays with shape
            ``(nconfig, 1, norb - 1)`` and index order ``[k, i, a]``.

    Notes
    -----
    This routine does not optimize ``lam`` or ``shift``.  Simultaneously
    applying ``lam[x, p] -> lam[x, p] + c[x]`` and
    ``shift[x] -> shift[x] + c[x]`` leaves all returned energies invariant.
    """
    phonon_configurations = lf_total_phonon_configurations(nmode=lam.shape[0], max_total=max_total)
    heff = lf_effective_one_body(tmat, g, omega, shift, lam)
    mo_energy, mo_coeff = np.linalg.eigh(heff)
    nocc = 1
    coeff = mo_coeff[:, 0]
    energy_hf = omega * np.dot(shift, shift) + mo_energy[0]
    site_couplings = lf_vacuum_coupling_site_matrices(tmat, g, omega, shift, lam, phonon_configurations)
    _, pure_matrix_elements, single_matrix_elements = lf_vacuum_coupling_mo_elements(
        site_couplings,
        mo_coeff,
        nocc,
    )
    total_occupations, pure_denominators, single_denominators = lf_mp2_denominators(mo_energy, omega, phonon_configurations, nocc)
    mp2_energy = lf_mp2_energy(pure_matrix_elements, single_matrix_elements, pure_denominators, single_denominators)
    total_energy = energy_hf + mp2_energy
    return {
        "hf_energy": energy_hf,
        "mp2_correction": mp2_energy,
        "total_energy": total_energy,
        "max_total": max_total,
        "coeff": coeff,
        "mo_energy": mo_energy,
        "mo_coeff": mo_coeff,
        "occupations": phonon_configurations,
        "total_occupations": total_occupations,
        "pure_matrix_elements": pure_matrix_elements,
        "pure_denominators": pure_denominators,
        "single_matrix_elements": single_matrix_elements,
        "single_denominators": single_denominators,
    }


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


def lf_hf_local_state(
    params: np.ndarray,
    tmat: np.ndarray,
    g: float,
    omega: float,
) -> tuple[float, np.ndarray, float]:
    """Evaluate the local-diagonal one-electron LF-HF ground state.

    Parameters
    ----------
    params
        Packed real parameters ``[ell[0:L], shift[0:L]]`` with shape
        ``(2 * norb,)``.  The conditional displacement is restricted to
        ``lam = diag(ell)``.
    tmat
        Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Local electron--phonon coupling strength.
    omega
        Positive phonon frequency.

    Returns
    -------
    total_energy
        LF-HF total energy including the global ``omega * shift @ shift``.
    coeff
        Normalized lowest eigenvector of the effective Hamiltonian, shape
        ``(norb,)``.
    orbital_energy
        Lowest eigenvalue of the effective Hamiltonian, excluding the global
        boson constant.
    """
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("Hopping matrix must be 2-dimensional and square.")
    norb = tmat.shape[0]
    if params.ndim != 1 or params.size != 2 * norb:
        raise ValueError("Parameter array must be 1-dimensional and of length 2*norb.")
    ell = params[:norb]
    shift = params[norb:]
    lam = np.diag(ell)
    heff = lf_effective_one_body(tmat, g, omega, shift, lam)
    eigvals, eigvecs = np.linalg.eigh(heff)
    orbital_energy = eigvals[0]
    coeff = eigvecs[:, 0]
    total_energy = omega * shift @ shift + orbital_energy
    return float(total_energy), coeff, float(orbital_energy)


def lf_hf_local_optimize(
    params0: np.ndarray,
    tmat: np.ndarray,
    g: float,
    omega: float,
    *,
    gtol: float = 1e-8,
    max_cycle: int = 500,
) -> scipy_optimize.OptimizeResult:
    """Optimize a one-electron local-diagonal LF-HF reference.

    The packed parameters have the order ``[ell[0:L], shift[0:L]]``, where
    ``lam = diag(ell)``.  A BFGS minimization with a three-point numerical
    gradient is applied to :func:`lf_hf_local_state`.

    Parameters
    ----------
    params0
        Finite real initial parameters with shape ``(2 * norb,)``.  The
        input array is not modified.
    tmat
        Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Local electron--phonon coupling strength.
    omega
        Positive phonon frequency.
    gtol
        Positive gradient-norm tolerance passed to BFGS.
    max_cycle
        Positive maximum number of BFGS iterations.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The BFGS result augmented with ``coeff``, ``orbital_energy``, and
        ``shift_residual``.  Its ``fun`` field is recomputed from the final
        parameters, and ``shift_residual`` is the infinity norm of the
        coherent-shift stationarity equation.
    """
    if params0.ndim != 1:
        raise ValueError("Initial parameter array must be 1-dimensional.")
    for param in params0:
        if not np.isfinite(param) or not np.isreal(param):
            raise ValueError("Initial parameter array must contain finite real values.")
    if gtol <= 0:
        raise ValueError("Gradient tolerance must be positive.")
    if max_cycle <= 0:
        raise ValueError("Maximum cycle count must be positive.")
    x0 = np.asarray(params0, dtype=np.float64, copy=True)
    lf_hf_local_state(x0, tmat, g, omega)

    def cost(params: np.ndarray) -> float:
        return lf_hf_local_state(params, tmat, g, omega)[0]

    result = scipy_optimize.minimize(
        cost,
        x0,
        method="BFGS",
        jac="3-point",
        options={"gtol": gtol, "maxiter": max_cycle, "disp": False},
    )
    final_energy, final_coeff, final_orbital_energy = lf_hf_local_state(result.x, tmat, g, omega)
    norb = tmat.shape[0]
    ell = result.x[:norb]
    shift = result.x[norb:]
    lam = np.diag(ell)
    stationary_shift = lf_stationary_shift(final_coeff, lam, g, omega)
    shift_residual = float(np.max(np.abs(shift - stationary_shift)))
    result["coeff"] = final_coeff
    result["orbital_energy"] = final_orbital_energy
    result["shift_residual"] = shift_residual
    result["fun"] = final_energy
    return result


def lf_hf_local_multistart(
    params0s: np.ndarray,
    tmat: np.ndarray,
    g: float,
    omega: float,
    *,
    gtol: float = 1e-8,
    max_cycle: int = 500,
) -> tuple[
    scipy_optimize.OptimizeResult,
    list[scipy_optimize.OptimizeResult],
]:
    """Optimize local-diagonal LF-HF from several explicit starting points.

    Parameters
    ----------
    params0s
        Finite real initial parameters with shape
        ``(nstart, 2 * norb)`` and ``nstart >= 1``.  Each row uses the
        packing ``[ell[0:L], shift[0:L]]`` and the array is not modified.
    tmat
        Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Local electron--phonon coupling strength.
    omega
        Positive phonon frequency.
    gtol
        Positive gradient-norm tolerance passed to every optimization.
    max_cycle
        Positive maximum number of BFGS iterations per starting point.

    Returns
    -------
    best
        Lowest-energy successful result.  Ties preserve the input order.
    results
        All optimization results in the same order as ``params0s``.

    Raises
    ------
    RuntimeError
        If none of the starting points converges successfully.
    """
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("Hopping matrix must be 2-dimensional and square.")
    if params0s.ndim != 2:
        raise ValueError("Initial parameter array must be 2-dimensional with shape (nstart, 2*norb).")
    norb = tmat.shape[0]
    nstart = params0s.shape[0]
    if params0s.shape[0] == 0:
        raise ValueError("Initial parameter array must have at least one starting point.")
    if params0s.shape[1] != 2 * norb:
        raise ValueError("Initial parameter array must have shape (nstart, 2*norb).")
    for i in range(nstart):
        for param in params0s[i]:
            if not np.isfinite(param) or not np.isreal(param):
                raise ValueError(f"Initial parameter array at index {i} must contain finite real values.")
    results = []
    for i in range(nstart):
        result = lf_hf_local_optimize(params0s[i], tmat, g, omega, gtol=gtol, max_cycle=max_cycle)
        results.append(result)
    successful_results = [res for res in results if res.success]
    if not successful_results:
        raise RuntimeError("All optimization attempts failed.")
    best = min(successful_results, key=lambda res: res.fun)
    return best, results


def lf_hf_full_multistart(
    mu0s: np.ndarray,
    tmat: np.ndarray,
    g: float,
    omega: float,
    *,
    gtol: float = 1e-8,
    max_cycle: int = 500,
) -> tuple[
    scipy_optimize.OptimizeResult,
    list[scipy_optimize.OptimizeResult],
]:
    """Optimize full-matrix one-electron LF-HF from explicit starts.

    Parameters
    ----------
    mu0s
        Finite real gauge-fixed displacements with shape
        ``(nstart, norb, norb)`` and index order ``[start, x, p]``.  At
        least one start is required, and the input array is not modified.
    tmat
        Square site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Local electron--phonon coupling strength.
    omega
        Positive phonon frequency.
    gtol
        Positive gradient-norm tolerance passed to every optimization.
    max_cycle
        Positive maximum number of BFGS iterations per starting point.

    Returns
    -------
    best
        Lowest-energy result among the optimizations for which
        ``result.success`` is true.  This is the same object as one entry
        in ``results``.
    results
        Optimization results in the same order as the supplied starts.

    Raises
    ------
    RuntimeError
        If every optimization reports failure.
    """
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("Hopping matrix must be 2-dimensional and square.")
    if mu0s.ndim != 3 or mu0s.shape[0] < 1 or mu0s.shape[1:] != tmat.shape:
        raise ValueError("Initial parameter array must be 3-dimensional with shape (nstart, norb, norb).")
    nstart = mu0s.shape[0]
    for mu in mu0s:
        if not np.all(np.isfinite(mu)) or not np.all(np.isreal(mu)):
            raise ValueError("Initial parameter array must contain finite real values.")
    results = []
    for k in range(nstart):
        result = lf_hf_full_optimize(mu0s[k], tmat, g, omega, gtol=gtol, max_cycle=max_cycle)
        results.append(result)
    successful_results = [res for res in results if res.success]
    if not successful_results:
        raise RuntimeError("All optimization attempts failed.")
    best = min(successful_results, key=lambda res: res.fun)
    return best, results


def lf_hf_local_initial_guesses(
    norb: int,
    g: float,
    omega: float,
    *,
    nrandom: int = 8,
    random_scale: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Generate reproducible CS-centered starts for local LF-HF.

    The first row is the uniform CS point with ``ell = 0`` and
    ``shift = -g / (omega * norb)``.  Every remaining row is an independent
    Gaussian perturbation of that point in the packed parameter order
    ``[ell[0:L], shift[0:L]]``.

    Parameters
    ----------
    norb
        Positive number of electronic sites/orbitals.
    g
        Finite real electron--phonon coupling strength.
    omega
        Finite positive phonon frequency.
    nrandom
        Nonnegative number of random starts in addition to the CS point.
    random_scale
        Finite positive standard deviation of every parameter perturbation.
    seed
        Seed passed to an independent :class:`numpy.random.Generator`.

    Returns
    -------
    numpy.ndarray
        Initial parameters with shape ``(1 + nrandom, 2 * norb)`` and
        ``float64`` dtype.  Row zero is exactly the CS point.
    """
    if not isinstance(norb, (int, np.integer)) or isinstance(norb, (bool, np.bool_)) or norb <= 0:
        raise ValueError("Number of orbitals must be a positive integer.")
    if not isinstance(nrandom, (int, np.integer)) or isinstance(nrandom, (bool, np.bool_)) or nrandom < 0:
        raise ValueError("Number of random guesses must be a non-negative integer.")
    if not np.isfinite(g) or not np.isreal(g):
        raise ValueError("Electron-phonon coupling g must be finite and real.")
    if not np.isfinite(omega) or not np.isreal(omega) or omega <= 0:
        raise ValueError("Phonon frequency omega must be finite, real, and positive.")
    if not np.isfinite(random_scale) or not np.isreal(random_scale) or random_scale <= 0:
        raise ValueError("Random scale must be finite, real, and positive.")
    params0s = np.zeros((nrandom + 1, 2 * norb), dtype=np.float64)
    params0s[0, norb:] = - g / (omega * norb)
    rng = np.random.default_rng(seed)
    for i in range(1, nrandom + 1):
        params0s[i, :norb] = rng.normal(loc=0.0, scale=random_scale, size=norb)
        params0s[i, norb:] = rng.normal(loc=- g / (omega * norb), scale=random_scale, size=norb)
    return params0s


def lf_hf_full_initial_guesses(
    norb: int,
    g: float,
    omega: float,
    *,
    nrandom: int = 8,
    random_scale: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Generate reproducible non-Gaussian starts for full-matrix LF-HF.

    The optimized variable is the gauge-fixed displacement
    ``mu[x, p] = lam[x, p] - shift[x]`` with ``shift = 0``.  Starts are
    returned in a fixed order: the coherent-state point, one localized
    polaron point for every site, and uniformly distributed perturbations
    around the coherent-state point.

    Parameters
    ----------
    norb
        Positive number of electronic sites/orbitals.  The local Holstein
        model has the same number of phonon modes.
    g
        Finite real electron--phonon coupling strength.
    omega
        Finite positive phonon frequency.
    nrandom
        Nonnegative number of uniform random starts appended after the
        deterministic starts.
    random_scale
        Finite positive dimensionless half-width of the uniform
        perturbations, whose dimensional scale is ``abs(g) / omega``.
    seed
        Seed passed to an independent :class:`numpy.random.Generator`.

    Returns
    -------
    numpy.ndarray
        Initial gauge-fixed displacements with shape
        ``(1 + norb + nrandom, norb, norb)``, ``float64`` dtype, and index
        order ``[start, x, p]``.  Row zero is the coherent-state point
        ``g / (omega * norb)``.  Row ``1 + s`` is zero except for
        ``mu[s, s] = g / omega``.
    """
    if not isinstance(norb, (int, np.integer)) or isinstance(norb, (bool, np.bool_)) or norb <= 0:
        raise ValueError("Number of orbitals must be a positive integer.")
    if not isinstance(nrandom, (int, np.integer)) or isinstance(nrandom, (bool, np.bool_)) or nrandom < 0:
        raise ValueError("Number of random guesses must be a non-negative integer.")
    if not np.isfinite(g) or not np.isreal(g):
        raise ValueError("Electron-phonon coupling g must be finite and real.")
    if not np.isfinite(omega) or not np.isreal(omega) or omega <= 0:
        raise ValueError("Phonon frequency omega must be finite, real, and positive.")
    if not np.isfinite(random_scale) or not np.isreal(random_scale) or random_scale <= 0:
        raise ValueError("Random scale must be finite, real, and positive.")
    mu_cs = np.full((norb, norb), g / (omega * norb), dtype=np.float64)
    guesses = np.empty((1 + norb + nrandom, norb, norb), dtype=np.float64)
    guesses[0] = mu_cs
    for s in range(norb):
        mu_s = np.zeros((norb, norb), dtype=np.float64)
        mu_s[s, s] = g / omega
        guesses[s + 1] = mu_s
    rng = np.random.default_rng(seed)
    for i in range(nrandom):
        perturbation = rng.uniform(-random_scale, random_scale, size=(norb, norb))
        guesses[norb + 1 + i] = mu_cs + (abs(g) / omega) * perturbation
    return guesses


def lf_hf_local_alpha_point(
    alpha: float,
    tmat: np.ndarray,
    omega: float,
    *,
    nrandom: int = 8,
    random_scale: float = 0.5,
    seed: int = 0,
    gtol: float = 1e-8,
    max_cycle: int = 500,
) -> tuple[
    scipy_optimize.OptimizeResult,
    list[scipy_optimize.OptimizeResult],
]:
    """Solve and diagnose one coupling point of local LF-HF.

    The dimensionless coupling is converted according to
    ``g = sqrt(alpha * omega)``.  Reproducible CS-centered starting points
    are optimized independently, and the lowest-energy successful result is
    augmented with its site density and density imbalance.

    Parameters
    ----------
    alpha
        Finite nonnegative coupling ``g**2 / omega``.
    tmat
        Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    omega
        Finite positive phonon frequency.
    nrandom
        Number of random starts in addition to the exact CS starting point.
    random_scale
        Standard deviation of the Gaussian starting-point perturbations.
    seed
        Seed used to generate the random starting points.
    gtol
        Gradient-norm tolerance passed to every local optimization.
    max_cycle
        Maximum number of BFGS iterations per starting point.

    Returns
    -------
    best
        Lowest-energy successful result, augmented with ``alpha``, ``g``,
        ``density``, ``density_imbalance``, ``nstart``, and ``nconverged``.
    results
        All local optimization results in starting-point order.  ``best`` is
        the same object as one entry of this list.
    """
    if not np.isfinite(alpha) or not np.isreal(alpha) or alpha < 0:
        raise ValueError("Alpha must be finite, real, and non-negative.")
    if not np.isfinite(omega) or not np.isreal(omega) or omega <= 0:
        raise ValueError("Phonon frequency omega must be finite, real, and positive.")
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("Hopping matrix must be 2-dimensional and square.")
    g = float(np.sqrt(alpha * omega))
    params0s = lf_hf_local_initial_guesses(
        tmat.shape[0],
        g,
        omega,
        nrandom=nrandom,
        random_scale=random_scale,
        seed=seed,
    )
    best, results = lf_hf_local_multistart(params0s, tmat, g, omega, gtol=gtol, max_cycle=max_cycle)
    density = cs_site_density(best["coeff"])
    best["alpha"] = float(alpha)
    best["g"] = float(g)
    best["density"] = density
    best["density_imbalance"] = float(np.max(density) - np.min(density))
    best["nstart"] = len(results)
    best["nconverged"] = sum(bool(res.success) for res in results)
    return best, results


def lf_hf_full_alpha_point(
    alpha: float,
    tmat: np.ndarray,
    omega: float,
    *,
    nrandom: int = 8,
    random_scale: float = 0.5,
    seed: int = 0,
    gtol: float = 1e-8,
    max_cycle: int = 500,
) -> tuple[
    scipy_optimize.OptimizeResult,
    list[scipy_optimize.OptimizeResult],
]:
    """Solve and diagnose one coupling point of full-matrix LF-HF.

    The dimensionless coupling is converted using
    ``g = sqrt(alpha * omega)``.  Gauge-fixed full-matrix starts are
    generated and optimized independently, after which the lowest-energy
    successful result is augmented with density and convergence diagnostics.

    Parameters
    ----------
    alpha
        Finite nonnegative coupling ``g**2 / omega``.
    tmat
        Square site-basis hopping matrix with shape ``(norb, norb)``.
    omega
        Finite positive phonon frequency.
    nrandom
        Number of uniform random starts in addition to the coherent-state
        point and one localized point per site.
    random_scale
        Dimensionless half-width of the uniform starting-point
        perturbations.
    seed
        Seed used by the independent random-number generator.
    gtol
        Positive gradient-norm tolerance passed to every optimization.
    max_cycle
        Positive maximum number of BFGS iterations per starting point.

    Returns
    -------
    best
        Lowest-energy successful result, augmented with ``alpha``, ``g``,
        ``density``, ``density_imbalance``, ``nstart``, and ``nconverged``.
        This is the same object as one entry in ``results``.
    results
        Optimization results in starting-point order.
    """
    if not np.isfinite(alpha) or not np.isreal(alpha) or alpha < 0:
        raise ValueError("Alpha must be finite, real, and non-negative.")
    if not np.isfinite(omega) or not np.isreal(omega) or omega <= 0:
        raise ValueError("Phonon frequency omega must be finite, real, and positive.")
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("Hopping matrix must be 2-dimensional and square.")
    g = float(np.sqrt(alpha * omega))
    mu0s = lf_hf_full_initial_guesses(
        tmat.shape[0],
        g,
        omega,
        nrandom=nrandom,
        random_scale=random_scale,
        seed=seed,
    )
    best, results = lf_hf_full_multistart(mu0s, tmat, g, omega, gtol=gtol, max_cycle=max_cycle)
    density = cs_site_density(best["coeff"])
    density_imbalance = float(np.max(density) - np.min(density))
    best["alpha"] = float(alpha)
    best["g"] = float(g)
    best["density"] = density
    best["density_imbalance"] = density_imbalance
    best["nstart"] = len(results)
    best["nconverged"] = sum(bool(res.success) for res in results)
    return best, results


def lf_mp2_alpha_point(
    alpha: float,
    tmat: np.ndarray,
    omega: float,
    *,
    max_total: int,
    nrandom: int = 8,
    random_scale: float = 0.5,
    seed: int = 0,
    gtol: float = 1e-8,
    max_cycle: int = 500,
) -> tuple[
    scipy_optimize.OptimizeResult,
    list[scipy_optimize.OptimizeResult],
]:
    """Optimize full-matrix LF-HF and evaluate LF-MP2 at one alpha.

    The lowest-energy successful full-matrix LF-HF result is used as the
    fixed reference for :func:`lf_mp2_reference_point`.  The MP2 result is
    added to that same :class:`scipy.optimize.OptimizeResult`, preserving its
    identity within the returned list of all starting-point results.

    Parameters
    ----------
    alpha
        Finite nonnegative dimensionless coupling ``g**2 / omega``.
    tmat
        Finite Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    omega
        Finite positive phonon frequency.
    max_total
        Positive inclusive cutoff on the total phonon occupation.  It is
        validated before any LF-HF optimizations are started.
    nrandom
        Number of random starts in addition to the coherent-state point and
        one localized point per site.
    random_scale
        Dimensionless half-width of the random starting-point perturbations.
    seed
        Seed used by the independent random-number generator.
    gtol
        Positive gradient-norm tolerance passed to every optimization.
    max_cycle
        Positive maximum number of BFGS iterations per starting point.

    Returns
    -------
    best
        The same lowest-energy successful optimization result returned by
        :func:`lf_hf_full_alpha_point`, augmented with all fields returned by
        :func:`lf_mp2_reference_point`.  ``best.fun`` remains the LF-HF
        energy, while ``best.total_energy`` is the LF-MP2 total energy.
    results
        All LF-HF optimization results in starting-point order.  ``best`` is
        the same object as one entry in this list.
    """
    if not isinstance(max_total, (int, np.integer)) or isinstance(max_total, (bool, np.bool_)):
        raise TypeError("max_total must be a positive integer")
    if max_total <= 0:
        raise ValueError("max_total must be a positive integer")
    best, results = lf_hf_full_alpha_point(
        alpha,
        tmat,
        omega,
        nrandom=nrandom,
        random_scale=random_scale,
        seed=seed,
        gtol=gtol,
        max_cycle=max_cycle,
    )
    mp2_result = lf_mp2_reference_point(tmat, best["g"], omega, best["shift"], best["lam"], max_total=max_total)
    best.update(mp2_result)
    return best, results


def lf_hf_full_state(
    lam: np.ndarray,
    shift: np.ndarray,
    tmat: np.ndarray,
    g: float,
    omega: float,
) -> tuple[float, np.ndarray, float]:
    """Evaluate a one-electron LF-HF state for a full LF parameter matrix.

    Parameters
    ----------
    lam
        Real conditional displacements with shape ``(nmode, norb)`` and
        index order ``lam[x, p]``.  The local Holstein model has one phonon
        mode per electronic site and therefore uses shape ``(L, L)``.
    shift
        Real coherent displacements ``z[x]`` with shape ``(nmode,)``.
    tmat
        Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Real local electron--phonon coupling strength.
    omega
        Positive phonon frequency.

    Returns
    -------
    total_energy
        LF-HF total energy including the global boson contribution
        ``omega * shift @ shift``.
    coeff
        Normalized lowest eigenvector of the zero-phonon effective
        one-electron Hamiltonian, with shape ``(norb,)``.
    orbital_energy
        Lowest eigenvalue of the effective one-electron Hamiltonian before
        adding the global boson contribution.

    Notes
    -----
    For one electron the simultaneous row shift
    ``lam[x, p] -> lam[x, p] + c[x]`` and ``shift[x] -> shift[x] + c[x]``
    leaves the total energy and electronic density invariant.
    """
    heff = lf_effective_one_body(tmat, g, omega, shift, lam)
    eigvals, eigvecs = np.linalg.eigh(heff)
    orbital_energy = eigvals[0]
    coeff = eigvecs[:, 0]
    total_energy = omega * shift @ shift + orbital_energy
    return float(total_energy), coeff, float(orbital_energy)


def lf_hf_full_optimize(
    mu0: np.ndarray,
    tmat: np.ndarray,
    g: float,
    omega: float,
    *,
    gtol: float = 1e-8,
    max_cycle: int = 500,
) -> scipy_optimize.OptimizeResult:
    """Optimize a one-electron full LF-HF reference.

    Parameters
    ----------
    mu0
        Finite real initial parameters with shape ``(nmode, norb)`` and
        index order ``mu0[x, p]``.  The input array is not modified.
    tmat
        Hermitian site-basis hopping matrix with shape ``(norb, norb)``.
    g
        Local electron--phonon coupling strength.
    omega
        Positive phonon frequency.
    gtol
        Positive gradient-norm tolerance passed to BFGS.
    max_cycle
        Positive maximum number of BFGS iterations.

    Returns
    -------
    scipy.optimize.OptimizeResult
        The BFGS result augmented with ``coeff``, ``orbital_energy``, and
        ``shift_residual``.  Its ``fun`` field is recomputed from the final
        parameters, and ``shift_residual`` is the infinity norm of the
        coherent-shift stationarity equation.
    """
    if mu0.ndim != 2 or mu0.shape[0] != mu0.shape[1]:
        raise ValueError("Initial parameter array must be 2-dimensional.")
    for mu_value in mu0.flat:
        if not np.isfinite(mu_value) or not np.isreal(mu_value):
            raise ValueError("Initial parameter array must contain finite real values.")
    if tmat.ndim != 2 or tmat.shape[0] != tmat.shape[1]:
        raise ValueError("Hopping matrix must be 2-dimensional and square.")
    if mu0.shape[0] != tmat.shape[0]:
        raise ValueError("Initial parameter array first dimension must match hopping matrix size.")
    if gtol <= 0:
        raise ValueError("Gradient tolerance must be positive.")
    if max_cycle <= 0:
        raise ValueError("Maximum cycle count must be positive.")
    norb = tmat.shape[0]
    x0 = np.asarray(mu0, dtype=np.float64, copy=True).ravel().copy()

    def cost(flat_mu: np.ndarray) -> float:
        mu = flat_mu.reshape((norb, norb))
        shift = np.zeros(norb)
        return lf_hf_full_state(mu, shift, tmat, g, omega)[0]

    result = scipy_optimize.minimize(
        cost,
        x0,
        method="BFGS",
        jac="3-point",
        options={"gtol": gtol, "maxiter": max_cycle, "disp": False},
    )
    final_shift = np.zeros(norb)
    final_mu = result.x.reshape((norb, norb)).copy()
    final_energy, final_coeff, final_orbital_energy = lf_hf_full_state(final_mu, final_shift, tmat, g, omega)
    stationary_shift = lf_stationary_shift(final_coeff, final_mu, g, omega)
    shift_residual = float(np.max(np.abs(final_shift - stationary_shift)))
    result["mu"] = final_mu
    result["lam"] = final_mu.copy()
    result["shift"] = final_shift
    result["coeff"] = final_coeff
    result["orbital_energy"] = final_orbital_energy
    result["shift_residual"] = shift_residual
    result["fun"] = final_energy
    return result


def lf_hf_multi_fixed_energy(
    tmat: np.ndarray,
    U: float,
    g: float,
    omega: float,
    nelec: tuple[int, int],
    occ_a: np.ndarray,
    occ_b: np.ndarray,
    lam: np.ndarray,
    shift: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Evaluate a fixed unrestricted LF-HF reference in the site basis.

    The local Hubbard--Holstein coupling is ``g * n[x] * (b[x] + b[x]^dag)``
    without a density offset. There is one phonon mode per site. This
    function evaluates supplied orbitals and full LF parameters; it does
    not optimize them.

    Parameters
    ----------
    tmat
        Real symmetric site-basis hopping matrix ``tmat[p, q]`` with shape
        ``(L, L)`` and dtype ``float64``.
    U, g, omega
        Finite real on-site repulsion, local electron--phonon coupling,
        and positive phonon frequency, respectively.
    nelec
        Fixed electron counts ``(neleca, nelecb)``, each between 0 and L.
    occ_a, occ_b
        Real orthonormal occupied-orbital columns ``occ_sigma[p, i]`` in
        the site basis, with shapes ``(L, neleca)`` and ``(L, nelecb)``
        and dtype ``float64``.
    lam
        Full real LF displacement matrix ``lam[x, p]``, with mode row x
        and site column p; shape ``(L, L)``, dtype ``float64``.
    shift
        Real coherent displacement ``shift[x]``; shape ``(L,)``, dtype
        ``float64``.

    Returns
    -------
    energy
        Total LF-HF expectation value for this fixed reference.
    spin_density
        Site density with shape ``(2, L)`` and index order
        ``spin_density[spin, p]``; spin 0 is alpha and spin 1 beta.

    Notes
    -----
    At fixed positive total electron number, adding the same ``c[x]`` to
    every ``lam[x, p]`` and ``N_e * c[x]`` to ``shift[x]`` leaves the
    represented state invariant. Fixing ``shift=0`` is one possible gauge
    for the full LF matrix.
    """
    if (not isinstance(tmat, np.ndarray) or tmat.ndim != 2
            or tmat.shape[0] == 0 or tmat.shape[0] != tmat.shape[1]):
        raise ValueError("tmat must have nonempty square shape (L, L)")
    L = tmat.shape[0]
    if (not isinstance(nelec, tuple) or len(nelec) != 2
            or any(isinstance(n, (bool, np.bool_))
                   or not isinstance(n, (int, np.integer))
                   or n < 0 or n > L for n in nelec)):
        raise ValueError("nelec must be a tuple (neleca, nelecb) with 0 <= n <= L")
    for name, array, shape in (
        ("tmat", tmat, (L, L)),
        ("occ_a", occ_a, (L, nelec[0])),
        ("occ_b", occ_b, (L, nelec[1])),
        ("lam", lam, (L, L)),
        ("shift", shift, (L,)),
    ):
        if (not isinstance(array, np.ndarray) or array.shape != shape
                or array.dtype != np.float64 or not np.all(np.isfinite(array))):
            raise ValueError(f"{name} must be a finite float64 array with shape {shape}")
    if not np.allclose(tmat, tmat.T, rtol=0.0, atol=1e-12):
        raise ValueError("tmat must be symmetric")
    for name, occupied in (("occ_a", occ_a), ("occ_b", occ_b)):
        if not np.allclose(occupied.T @ occupied,
                           np.eye(occupied.shape[1]), rtol=0.0, atol=1e-10):
            raise ValueError(f"{name} columns must be orthonormal")
    for name, value in (("U", U), ("g", g), ("omega", omega)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)):
            raise ValueError(f"{name} must be a finite real scalar")
    if omega <= 0:
        raise ValueError("omega must be positive")
    D_a = occ_a @ occ_a.T
    D_b = occ_b @ occ_b.T
    rho_a = np.diag(D_a)
    rho_b = np.diag(D_b)
    rho = rho_a + rho_b
    rho_joint = np.stack((rho_a, rho_b), axis=0)
    n_orb = tmat.shape[0]
    Q = np.zeros((n_orb, n_orb))
    for p in range(n_orb):
        for q in range(n_orb):
            Q[p, q] = rho[p] * rho[q] - D_a[p, q] * D_a[q, p] - D_b[p, q] * D_b[q, p] + (rho[p] if p == q else 0)
    S = np.zeros((n_orb, n_orb))
    for p in range(n_orb):
        for q in range(n_orb):
            S[p, q] = np.exp(-0.5 * np.sum((lam[:,p]-lam[:,q])**2))
    E_t = np.sum(tmat * S * (D_a + D_b).T)
    E_U = U * np.sum(rho_a * rho_b)
    E_ph = omega * (shift.T @ shift - 2 * shift.T @ lam @ rho + np.trace(lam @ Q @ lam.T))
    E_ep = 2 * g * np.sum(shift*rho - np.sum(lam * Q, axis=1))
    E = E_t + E_U + E_ph + E_ep
    return E, rho_joint


def lf_hf_multi_fixed_fock(
    tmat: np.ndarray,
    U: float,
    g: float,
    omega: float,
    nelec: tuple[int, int],
    occ_a: np.ndarray,
    occ_b: np.ndarray,
    lam: np.ndarray,
    shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build spin-resolved site-basis Fock matrices for a fixed LF reference.

    This is the derivative of ``lf_hf_multi_fixed_energy`` with respect to
    each spin density matrix while ``lam`` and ``shift`` remain fixed. It
    uses the paper's uncentered local Hubbard--Holstein coupling and one
    phonon mode per site. It does not optimize the occupied orbitals.

    Parameters
    ----------
    tmat
        Real symmetric site-basis hopping matrix ``tmat[p, q]``, shape
        ``(L, L)``, dtype ``float64``.
    U, g, omega
        Finite real on-site repulsion and local coupling, followed by a
        positive phonon frequency.
    nelec
        Fixed counts ``(neleca, nelecb)``, each between 0 and L.
    occ_a, occ_b
        Real orthonormal occupied-orbital columns ``occ_sigma[p, i]`` in
        the site basis; shapes ``(L, neleca)`` and ``(L, nelecb)`` and
        dtype ``float64``.
    lam
        Full real LF matrix ``lam[x, p]`` with mode row and site column;
        shape ``(L, L)``, dtype ``float64``.
    shift
        Real coherent displacement ``shift[x]``, shape ``(L,)``, dtype
        ``float64``.

    Returns
    -------
    F_a, F_b
        Real symmetric Fock matrices for alpha and beta electrons, each
        with shape ``(L, L)`` and site index order ``[p, q]``.
    """
    if (not isinstance(tmat, np.ndarray) or tmat.ndim != 2
            or tmat.shape[0] == 0 or tmat.shape[0] != tmat.shape[1]):
        raise ValueError("tmat must have nonempty square shape (L, L)")
    L = tmat.shape[0]
    if (not isinstance(nelec, tuple) or len(nelec) != 2
            or any(isinstance(n, (bool, np.bool_))
                   or not isinstance(n, (int, np.integer))
                   or n < 0 or n > L for n in nelec)):
        raise ValueError("nelec must be a tuple (neleca, nelecb) with 0 <= n <= L")
    for name, array, shape in (
        ("tmat", tmat, (L, L)),
        ("occ_a", occ_a, (L, nelec[0])),
        ("occ_b", occ_b, (L, nelec[1])),
        ("lam", lam, (L, L)),
        ("shift", shift, (L,)),
    ):
        if (not isinstance(array, np.ndarray) or array.shape != shape
                or array.dtype != np.float64 or not np.all(np.isfinite(array))):
            raise ValueError(f"{name} must be a finite float64 array with shape {shape}")
    if not np.allclose(tmat, tmat.T, rtol=0.0, atol=1e-12):
        raise ValueError("tmat must be symmetric")
    for name, occupied in (("occ_a", occ_a), ("occ_b", occ_b)):
        if not np.allclose(occupied.T @ occupied,
                           np.eye(occupied.shape[1]), rtol=0.0, atol=1e-10):
            raise ValueError(f"{name} columns must be orthonormal")
    for name, value in (("U", U), ("g", g), ("omega", omega)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)):
            raise ValueError(f"{name} must be a finite real scalar")
    if omega <= 0:
        raise ValueError("omega must be positive")
    D_a = occ_a @ occ_a.T
    D_b = occ_b @ occ_b.T
    rho = np.diag(D_a) + np.diag(D_b)
    G = lam.T @ lam
    d = np.diag(G)
    S = np.exp(-0.5 * (d[:, None] + d[None, :] - 2 * G))
    h = tmat * S
    A = omega * G - g * (lam + lam.T)
    v = 2 * g * shift - 2 * omega * lam.T @ shift
    F_a = h + np.diag(v + np.diag(A) + 2 * A @ rho + U * np.diag(D_b)) - 2 * A * D_a
    F_b = h + np.diag(v + np.diag(A) + 2 * A @ rho + U * np.diag(D_a)) - 2 * A * D_b
    return F_a, F_b


def lf_hf_multi_optimize(
    tmat: np.ndarray,
    U: float,
    g: float,
    omega: float,
    nelec: tuple[int, int],
    mo_coeff_a0: np.ndarray,
    mo_coeff_b0: np.ndarray,
    lam0: np.ndarray,
    *,
    gtol: float = 1e-6,
    max_cycle: int = 1000,
) -> scipy_optimize.OptimizeResult:
    """Jointly optimize unrestricted orbitals and the full LF matrix.

    This is the unrestricted LF-HF variational problem. The coherent shift is
    fixed to zero. Occupied orbitals are the first ``nelec[spin]`` columns
    of each full site-basis orbital matrix. Neither the input orbitals nor
    ``lam0`` are modified.

    Parameters
    ----------
    tmat
        Real symmetric site-basis hopping matrix, shape ``(L, L)``,
        dtype ``float64``.
    U, g, omega
        Finite real Hubbard repulsion and local coupling, followed by a
        positive phonon frequency.
    nelec
        Electron counts ``(neleca, nelecb)``, each between zero and L.
    mo_coeff_a0, mo_coeff_b0
        Initial full real orthogonal orbital matrices, each with shape
        ``(L, L)`` and dtype ``float64``; row is site, column is MO.
    lam0
        Initial full LF matrix ``lam0[x, p]``, shape ``(L, L)`` and
        dtype ``float64``; x is phonon mode, p is electronic site.
    gtol
        Positive BFGS gradient tolerance for the joint parameter vector.
    max_cycle
        Positive maximum number of BFGS iterations per start.

    Returns
    -------
    scipy.optimize.OptimizeResult
        ``fun`` is the variational LF-HF energy. ``mo_coeff`` and
        ``mo_energy`` have shapes ``(2, L, L)`` and ``(2, L)`` with alpha
        before beta; occupied MOs precede virtual MOs. ``mo_occ`` has
        shape ``(2, L)`` and contains zeros and ones. ``lam`` has shape
        ``(L, L)``, ``shift`` is the zero vector of length L, and
        ``spin_density`` has shape ``(2, L)``. Also return ``nit``,
        ``orbital_grad_norm``, ``lam_grad_max``, ``success``, and ``message``.
        A numerical optimizer status alone does not establish success:
        the final physical orbital and LF parameter residuals must pass.
    """
    if (not isinstance(tmat, np.ndarray) or tmat.ndim != 2
            or tmat.shape[0] == 0 or tmat.shape[0] != tmat.shape[1]):
        raise ValueError("tmat must have nonempty square shape (L, L)")
    L = tmat.shape[0]
    if (not isinstance(nelec, tuple) or len(nelec) != 2
            or any(isinstance(n, (bool, np.bool_))
                   or not isinstance(n, (int, np.integer))
                   or n < 0 or n > L for n in nelec)):
        raise ValueError("nelec must be a tuple (neleca, nelecb) with 0 <= n <= L")
    for name, coeff in (("mo_coeff_a0", mo_coeff_a0),
                        ("mo_coeff_b0", mo_coeff_b0)):
        if (not isinstance(coeff, np.ndarray) or coeff.shape != (L, L)
                or coeff.dtype != np.float64
                or not np.all(np.isfinite(coeff))):
            raise ValueError(f"{name} must be a finite float64 array with shape {(L, L)}")
        if not np.allclose(coeff.T @ coeff, np.eye(L), rtol=0.0, atol=1e-10):
            raise ValueError(f"{name} columns must be orthonormal")
    # Reuse the accepted fixed-reference checks for model data and lam0.
    lf_hf_multi_fixed_fock(
        tmat, U, g, omega, nelec,
        mo_coeff_a0[:, :nelec[0]], mo_coeff_b0[:, :nelec[1]],
        lam0, np.zeros(L, dtype=np.float64),
    )
    if (isinstance(gtol, (bool, np.bool_))
            or not isinstance(gtol, (int, float, np.integer, np.floating))
            or not np.isfinite(gtol) or gtol <= 0):
        raise ValueError("gtol must be a finite positive real scalar")
    if (isinstance(max_cycle, (bool, np.bool_))
            or not isinstance(max_cycle, (int, np.integer)) or max_cycle <= 0):
        raise ValueError("max_cycle must be a positive integer")
    na, nb = nelec
    nrot_a = na * (L - na)
    nrot_b = nb * (L - nb)
    x0 = np.r_[np.zeros(nrot_a + nrot_b), lam0.ravel()]

    def unpack(x):
        kappa_a = x[:nrot_a].reshape(L - na, na)
        kappa_b = x[nrot_a:nrot_a + nrot_b].reshape(L - nb, nb)
        lam = x[nrot_a + nrot_b:].reshape(L, L)

        K_a = np.zeros((L, L), dtype=np.float64)
        K_b = np.zeros((L, L), dtype=np.float64)
        K_a[na:, :na] = kappa_a
        K_a[:na, na:] = -kappa_a.T
        K_b[nb:, :nb] = kappa_b
        K_b[:nb, nb:] = -kappa_b.T

        # A line search can sample very large rotations. Polar projection
        # removes accumulated matrix-exponential roundoff without changing
        # the exact orthogonal rotation represented by exp(K).
        rotation_a, _ = scipy_linalg.polar(scipy_linalg.expm(K_a))
        rotation_b, _ = scipy_linalg.polar(scipy_linalg.expm(K_b))
        C_a = mo_coeff_a0 @ rotation_a
        C_b = mo_coeff_b0 @ rotation_b
        return C_a, C_b, lam

    shift = np.zeros(L, dtype=np.float64)

    def cost(x):
        try:
            C_a, C_b, lam = unpack(x)
            energy, _ = lf_hf_multi_fixed_energy(
                tmat, U, g, omega, nelec,
                C_a[:, :na], C_b[:, :nb], lam, shift,
            )
        except (ValueError, FloatingPointError):
            return 1e100
        return energy if np.isfinite(energy) else 1e100

    result = scipy_optimize.minimize(
        cost, x0, method="BFGS", jac="3-point",
        options={"gtol": gtol, "maxiter": max_cycle},
    )
    C_a, C_b, lam = unpack(result.x)
    energy, spin_density = lf_hf_multi_fixed_energy(
        tmat, U, g, omega, nelec, C_a[:, :na], C_b[:, :nb], lam, shift,
    )
    F_a, F_b = lf_hf_multi_fixed_fock(
        tmat, U, g, omega, nelec, C_a[:, :na], C_b[:, :nb], lam, shift,
    )
    D_a = C_a[:, :na] @ C_a[:, :na].T
    D_b = C_b[:, :nb] @ C_b[:, :nb].T
    orbital_grad_norm = np.hypot(
        np.linalg.norm(F_a @ D_a - D_a @ F_a),
        np.linalg.norm(F_b @ D_b - D_b @ F_b),
    )

    rho = np.diag(D_a + D_b)
    Q = np.outer(rho, rho) - D_a * D_a - D_b * D_b + np.diag(rho)
    G = lam.T @ lam
    diag_G = np.diag(G)
    S = np.exp(-0.5 * (diag_G[:, None] + diag_G[None, :] - 2 * G))
    W = tmat * S * (D_a + D_b)
    lam_gradient = (-2 * (lam * W.sum(axis=1)[None, :] - lam @ W)
                    + 2 * omega * lam @ Q - 2 * g * Q)
    lam_grad_max = float(np.max(np.abs(lam_gradient)))

    def canonicalize(coeff, fock, nocc):
        fock_mo = coeff.T @ fock @ coeff
        occupied_energy, occupied_rotation = np.linalg.eigh(
            fock_mo[:nocc, :nocc]
        )
        virtual_energy, virtual_rotation = np.linalg.eigh(
            fock_mo[nocc:, nocc:]
        )
        rotation = scipy_linalg.block_diag(
            occupied_rotation, virtual_rotation
        )
        return coeff @ rotation, np.r_[occupied_energy, virtual_energy]

    C_a, eps_a = canonicalize(C_a, F_a, na)
    C_b, eps_b = canonicalize(C_b, F_b, nb)
    energy_check, spin_density_check = lf_hf_multi_fixed_energy(
        tmat, U, g, omega, nelec, C_a[:, :na], C_b[:, :nb], lam, shift,
    )
    energy_consistent = bool(
        np.isfinite(energy_check)
        and abs(energy_check - result.fun) <= 1e-9
        and np.allclose(spin_density_check, spin_density, atol=1e-10, rtol=0)
    )
    residual_ok = bool(
        np.isfinite(orbital_grad_norm) and np.isfinite(lam_grad_max)
        and orbital_grad_norm <= 1e-5 and lam_grad_max <= 1e-5
    )
    result.optimizer_success = bool(result.success)
    result.residual_ok = residual_ok
    result.energy_consistent = energy_consistent
    result.success = bool(result.optimizer_success and residual_ok
                          and energy_consistent)
    result.mo_coeff = np.stack((C_a, C_b))
    result.mo_energy = np.stack((eps_a, eps_b))
    result.mo_occ = np.stack((np.arange(L) < na, np.arange(L) < nb)).astype(np.float64)
    result.lam = lam.copy()
    result.shift = shift.copy()
    result.spin_density = spin_density_check
    result.orbital_grad_norm = float(orbital_grad_norm)
    result.lam_grad_max = lam_grad_max
    return result



def lf_mp2_unrestricted_reference_point(
    tmat: np.ndarray,
    U: float,
    g: float,
    omega: float,
    nelec: tuple[int, int],
    mo_coeff: np.ndarray,
    mo_energy: np.ndarray,
    lam: np.ndarray,
    shift: np.ndarray,
    *,
    max_total: int,
    max_excited_modes: int = 2,
    denominator_tol: float = 1e-10,
) -> dict[str, object]:
    """Evaluate LF-MP2 on a fixed unrestricted LF-HF reference.

    Supports any positive number of sites and any alpha/beta electron counts.
    The local model has one phonon mode per site. For a non-vacuum phonon
    configuration the transformed Hamiltonian couples the reference only to
    itself and electronic singles; zero-phonon doubles arise from the static
    transformed density interaction. All orbital arrays are real float64.

    Parameters
    ----------
    tmat, U, g, omega, nelec
        Model data in the same convention as lf_hf_multi_fixed_energy.
    mo_coeff
        Canonical full alpha/beta orbital matrices, shape (2, L, L), with
        occupied orbitals before virtual orbitals.
    mo_energy
        Corresponding spin orbital energies, shape (2, L).
    lam, shift
        Full LF matrix of shape (L, L) and coherent displacement of shape
        (L,). The gauge-fixed optimizer returns a zero shift.
    max_total
        Inclusive total phonon occupation cutoff. Polar's nph=9 means 8.
    max_excited_modes
        Maximum number of modes with nonzero occupation; 2 follows Polar.
    denominator_tol
        Any excitation energy at or below this positive threshold raises
        ValueError instead of entering the MP2 sum.

    Returns
    -------
    dict
        HF, separate MP2 channel corrections, total energy, phonon
        configurations, smallest positive denominator, and channel counts.
    """
    if not isinstance(tmat, np.ndarray) or tmat.ndim != 2:
        raise ValueError("tmat must be a square float64 array")
    L = tmat.shape[0]
    if (not isinstance(mo_coeff, np.ndarray)
            or mo_coeff.shape != (2, L, L)
            or mo_coeff.dtype != np.float64
            or not np.all(np.isfinite(mo_coeff))):
        raise ValueError("mo_coeff must be finite float64 with shape (2, L, L)")
    if (not isinstance(mo_energy, np.ndarray)
            or mo_energy.shape != (2, L)
            or mo_energy.dtype != np.float64
            or not np.all(np.isfinite(mo_energy))):
        raise ValueError("mo_energy must be finite float64 with shape (2, L)")
    for coeff in mo_coeff:
        if not np.allclose(coeff.T @ coeff, np.eye(L), atol=1e-10, rtol=0):
            raise ValueError("mo_coeff columns must be orthonormal")
    if (isinstance(max_total, (bool, np.bool_))
            or not isinstance(max_total, (int, np.integer))
            or max_total < 1):
        raise ValueError("max_total must be a positive integer")
    if (isinstance(max_excited_modes, (bool, np.bool_))
            or not isinstance(max_excited_modes, (int, np.integer))
            or max_excited_modes < 1):
        raise ValueError("max_excited_modes must be a positive integer")
    if (isinstance(denominator_tol, (bool, np.bool_))
            or not isinstance(denominator_tol, (int, float, np.integer, np.floating))
            or not np.isfinite(denominator_tol) or denominator_tol <= 0):
        raise ValueError("denominator_tol must be finite and positive")

    if (not isinstance(nelec, tuple) or len(nelec) != 2
            or any(isinstance(n, (bool, np.bool_))
                   or not isinstance(n, (int, np.integer))
                   or n < 0 or n > L for n in nelec)):
        raise ValueError("nelec must be two electron counts between zero and L")
    na, nb = nelec
    hf_energy, _ = lf_hf_multi_fixed_energy(
        tmat, U, g, omega, nelec,
        mo_coeff[0, :, :na], mo_coeff[1, :, :nb], lam, shift,
    )
    fock = lf_hf_multi_fixed_fock(
        tmat, U, g, omega, nelec,
        mo_coeff[0, :, :na], mo_coeff[1, :, :nb], lam, shift,
    )
    fock_mo = np.stack([
        mo_coeff[s].T @ fock[s] @ mo_coeff[s] for s in range(2)
    ])
    for s, nocc in enumerate(nelec):
        for block in (fock_mo[s, :nocc, :nocc],
                      fock_mo[s, nocc:, nocc:]):
            if np.max(np.abs(block - np.diag(np.diag(block))), initial=0.0) > 1e-7:
                raise ValueError("occupied and virtual Fock blocks must be canonical")
        if np.max(np.abs(np.diag(fock_mo[s]) - mo_energy[s]), initial=0.0) > 1e-7:
            raise ValueError("mo_energy must match the canonical Fock diagonal")

    # For one electron the exact zero-phonon Hamiltonian defines the
    # perturbative orbital energies, as in the existing Fig. 2b routine.
    # Its virtual spectrum can differ from the unrestricted Fock spectrum.
    mo_coeff = mo_coeff.copy()
    mo_energy = mo_energy.copy()
    fock_mo = fock_mo.copy()
    if na + nb == 1:
        active_spin = 0 if na else 1
        effective = lf_effective_one_body(tmat, g, omega, shift, lam)
        effective_energy, effective_coeff = np.linalg.eigh(effective)
        overlap = abs(np.dot(effective_coeff[:, 0], mo_coeff[active_spin, :, 0]))
        if overlap < 1 - 1e-5:
            raise ValueError("one-electron reference is not the lowest effective orbital")
        mo_coeff[active_spin] = effective_coeff
        mo_energy[active_spin] = effective_energy
        fock_mo[active_spin] = np.diag(effective_energy)

    occupations = lf_total_phonon_configurations(L, max_total)
    occupations = occupations[
        np.count_nonzero(occupations, axis=1) <= max_excited_modes
    ]
    totals = occupations.sum(axis=1)
    site_couplings = lf_vacuum_coupling_site_matrices(
        tmat, g, omega, shift, lam, occupations,
    )
    mo_couplings = np.stack([
        np.einsum("pi,kpq,qj->kij", mo_coeff[s], site_couplings,
                  mo_coeff[s], optimize=True)
        for s in range(2)
    ])
    pure_amplitudes = np.zeros(len(occupations), dtype=np.float64)
    phonon_single_correction = 0.0
    zero_single_correction = 0.0
    minimum_denominator = float(omega)
    single_count = 0

    for s, nocc in enumerate(nelec):
        pure_amplitudes += np.trace(
            mo_couplings[s, :, :nocc, :nocc], axis1=1, axis2=2,
        )
        virtual_energies = mo_energy[s, nocc:]
        occupied_energies = mo_energy[s, :nocc]
        gap = virtual_energies[None, :] - occupied_energies[:, None]
        if gap.size:
            smallest_gap = float(np.min(gap))
            minimum_denominator = min(minimum_denominator, smallest_gap)
            if smallest_gap <= denominator_tol:
                raise ValueError(
                    f"nonpositive or near-zero electronic single denominator: {smallest_gap}"
                )
            phonon_denominator = gap[None, :, :] + omega * totals[:, None, None]
            amplitudes = mo_couplings[s, :, nocc:, :nocc].transpose(0, 2, 1)
            phonon_single_correction -= float(
                np.sum(np.abs(amplitudes)**2 / phonon_denominator)
            )
            zero_amplitudes = fock_mo[s, nocc:, :nocc].T
            zero_single_correction -= float(
                np.sum(np.abs(zero_amplitudes)**2 / gap)
            )
            single_count += nocc * (L - nocc)

    phonon_pure_correction = -float(
        np.sum(np.abs(pure_amplitudes)**2 / (omega * totals))
    )

    # Site determinant vectors provide the static two-body matrix elements
    # without spin-dependent exchange or excitation-sign bookkeeping.
    site_strings = [
        (np.empty((1, 0), dtype=np.int64) if n == 0 else
         np.asarray(list(combinations(range(L), n)), dtype=np.int64).reshape(-1, n))
        for n in nelec
    ]
    site_bits = []
    for strings in site_strings:
        bits = np.zeros((len(strings), L), dtype=np.float64)
        bits[np.arange(len(strings))[:, None], strings] = 1.0
        site_bits.append(bits)
    nsite = site_bits[0][:, None, :] + site_bits[1][None, :, :]
    G = lam.T @ lam
    A = omega * G - g * (lam + lam.T)
    site_potential = (
        np.einsum("abp,pq,abq->ab", nsite, A, nsite, optimize=True)
        + U * (site_bits[0] @ site_bits[1].T)
    )

    def determinant_vector(spin, occupied_mos):
        nocc = nelec[spin]
        if nocc == 0:
            return np.ones(1, dtype=np.float64)
        coeff = mo_coeff[spin]
        return np.linalg.det(
            coeff[site_strings[spin]][:, :, occupied_mos]
        )

    references = [
        determinant_vector(s, tuple(range(nelec[s]))) for s in range(2)
    ]
    transitions = [{tuple(range(nelec[s])): references[s]**2}
                   for s in range(2)]

    def transition(spin, occupied_mos):
        occupied_mos = tuple(occupied_mos)
        if occupied_mos not in transitions[spin]:
            transitions[spin][occupied_mos] = (
                determinant_vector(spin, occupied_mos) * references[spin]
            )
        return transitions[spin][occupied_mos]

    def excited_occ(spin, holes, particles):
        return tuple(sorted(
            (set(range(nelec[spin])) - set(holes)) | set(particles)
        ))

    def double_term(holes_a, particles_a, holes_b, particles_b):
        delta = (
            np.sum(mo_energy[0, list(particles_a)])
            - np.sum(mo_energy[0, list(holes_a)])
            + np.sum(mo_energy[1, list(particles_b)])
            - np.sum(mo_energy[1, list(holes_b)])
        )
        return float(delta), float(np.einsum(
            "a,ab,b->", transition(0, excited_occ(0, holes_a, particles_a)),
            site_potential, transition(1, excited_occ(1, holes_b, particles_b)),
            optimize=True,
        ))

    zero_double_correction = 0.0
    double_count = 0
    for s, nocc in enumerate(nelec):
        for holes in combinations(range(nocc), 2):
            for particles in combinations(range(nocc, L), 2):
                args = (holes, particles, (), ()) if s == 0 else (
                    (), (), holes, particles
                )
                denominator, amplitude = double_term(*args)
                minimum_denominator = min(minimum_denominator, denominator)
                if denominator <= denominator_tol:
                    raise ValueError(
                        f"nonpositive or near-zero electronic double denominator: {denominator}"
                    )
                zero_double_correction -= amplitude**2 / denominator
                double_count += 1
    for hole_a in range(na):
        for particle_a in range(na, L):
            for hole_b in range(nb):
                for particle_b in range(nb, L):
                    denominator, amplitude = double_term(
                        (hole_a,), (particle_a,), (hole_b,), (particle_b,)
                    )
                    minimum_denominator = min(minimum_denominator, denominator)
                    if denominator <= denominator_tol:
                        raise ValueError(
                            f"nonpositive or near-zero electronic double denominator: {denominator}"
                        )
                    zero_double_correction -= amplitude**2 / denominator
                    double_count += 1

    mp2_correction = (
        phonon_pure_correction + phonon_single_correction
        + zero_single_correction + zero_double_correction
    )
    return {
        "hf_energy": float(hf_energy),
        "mp2_correction": float(mp2_correction),
        "total_energy": float(hf_energy + mp2_correction),
        "phonon_pure_correction": phonon_pure_correction,
        "phonon_single_correction": phonon_single_correction,
        "zero_single_correction": zero_single_correction,
        "zero_double_correction": float(zero_double_correction),
        "occupations": occupations,
        "max_total": int(max_total),
        "max_excited_modes": int(max_excited_modes),
        "minimum_denominator": float(minimum_denominator),
        "zero_single_count": single_count,
        "zero_double_count": double_count,
    }
