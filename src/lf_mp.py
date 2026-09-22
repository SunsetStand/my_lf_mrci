"""Lang--Firsov mean-field reference and perturbative corrections."""

import math

import numpy as np
from scipy import optimize as scipy_optimize

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
