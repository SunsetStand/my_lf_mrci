import numpy as np

from src import my_direct_ep


def lf_frame_overlap(
    lam: np.ndarray,
    shift: np.ndarray,
    nelec: tuple[int,int] = (1,0),
) -> np.ndarray:
    """Return the physical-state overlap for one-electron LF frames.

    Frames use real conditional coherent displacements
    ``eta[A, x, p] = shift[A, x] - lam[A, x, p]``.  The result is
    ``S[A, p, B, q] = delta[p, q] * exp(-||eta[A, :, p] -
    eta[B, :, q]||**2 / 2)``.  Different frames may be linearly
    dependent, so the flattened overlap need not be invertible.

    Parameters
    ----------
    lam
        Finite float64 LF parameters with shape ``(K, L, L)`` and
        index order ``[frame, phonon mode, electron site]``.
    shift
        Finite float64 coherent shifts with shape ``(K, L)``
        and index order ``[frame, phonon mode]``.
    nelec
        Reserved for a future electron-sector extension.  This
        implementation describes only the single-alpha-electron
        site basis and does not inspect this argument.

    Returns
    -------
    numpy.ndarray
        Float64 overlap tensor with shape ``(K, L, K, L)`` and
        index order ``[A, p, B, q]``.  At a solver boundary it can
        be reshaped to ``(K*L, K*L)`` with ``a = A*L + p``.

    Raises
    ------
    ValueError
        If an array has incompatible or empty dimensions, or
        contains nonfinite values.
    TypeError
        If an array does not have float64 dtype.
    """
    if lam.ndim != 3 or lam.shape[1] != lam.shape[2]:
        raise ValueError("lam must be a 3D array with shape (n, m, m)")
    K, L, _ = lam.shape
    if K <= 0 or L <= 0:
        raise ValueError("lam must have positive dimensions")
    if shift.ndim != 2 or shift.shape != (K, L):
        raise ValueError("shift must be a 2D array with shape (n, m)")
    if lam.dtype != np.float64 or shift.dtype != np.float64:
        raise TypeError("lam and shift must have float64 dtype")
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(shift)):
        raise ValueError("lam and shift must contain only finite values")
    eta = shift[:, :, None] - lam
    S = np.zeros((K,L,K,L), dtype=np.float64)
    for A in range(K):
        for B in range(K):
            for p in range(L):
                S[A,p,B,p] = np.exp(-np.sum((eta[A,:,p] - eta[B,:,p])**2)/2)
    return S


def lf_frame_hamiltonian(
    tmat: np.ndarray,
    g: float,
    omega: float,
    lam: np.ndarray,
    shift: np.ndarray,
    nelec: tuple[int,int] = (1, 0),
) -> np.ndarray:
    """Return the physical Hamiltonian kernel between one-electron LF frames.

    With ``eta[A, x, p] = shift[A, x] - lam[A, x, p]``, each element
    ``H[A, p, B, q]`` is the matrix element of the uncentered
    Hubbard-Holstein Hamiltonian between ``|p>|eta[A, :, p]>`` and
    ``|q>|eta[B, :, q]>``.  The phonon energy sums over every mode;
    the local electron-phonon coupling uses only mode ``p`` when
    ``p == q``.  The boson Hamiltonian has no zero-point constant.

    Parameters
    ----------
    tmat
        Finite real symmetric hopping matrix with float64 dtype and
        shape ``(L, L)``, indexed as ``[p, q]``.  Diagonal entries
        are allowed.
    g
        Finite real local electron-phonon coupling.
    omega
        Finite positive phonon frequency shared by the L modes.
    lam
        Finite float64 LF parameters with shape ``(K, L, L)`` and
        index order ``[frame, phonon mode, electron site]``.
    shift
        Finite float64 coherent shifts with shape ``(K, L)`` and
        index order ``[frame, phonon mode]``.
    nelec
        Reserved for a future electron-sector extension.  This
        implementation describes only the single-alpha-electron
        site basis and does not inspect this argument.

    Returns
    -------
    numpy.ndarray
        Float64 Hamiltonian tensor with shape ``(K, L, K, L)`` and
        index order ``[A, p, B, q]``.  At a solver boundary it can
        be reshaped to ``(K*L, K*L)`` with ``a = A*L + p``.  This
        matrix is paired with the nonorthogonal overlap from
        ``lf_frame_overlap``.

    Raises
    ------
    ValueError
        If dimensions are incompatible or empty, arrays contain
        nonfinite values, ``tmat`` is not symmetric, or the scalar
        parameters violate their finite real and positive-frequency
        requirements.
    TypeError
        If an array does not have float64 dtype.
    """
    if lam.ndim != 3 or lam.shape[1] != lam.shape[2]:
        raise ValueError("lam must be a 3D array with shape (n, m, m)")
    K, L, _ = lam.shape
    if K <= 0 or L <= 0:
        raise ValueError("lam must have positive dimensions")
    if shift.ndim != 2 or shift.shape != (K, L):
        raise ValueError("shift must be a 2D array with shape (n, m)")
    if tmat.ndim != 2 or tmat.shape != (L, L):
        raise ValueError("tmat must be a 2D array with shape (m, m)")
    if lam.dtype != np.float64 or shift.dtype != np.float64 or tmat.dtype != np.float64:
        raise TypeError("lam, shift, and tmat must have float64 dtype")
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(shift)) or not np.all(np.isfinite(tmat)):
        raise ValueError("lam, shift, and tmat must contain only finite values")
    if not np.array_equal(tmat, tmat.T):
        raise ValueError("tmat must be symmetric")
    if not np.isfinite(g) or not np.isreal(g):
        raise ValueError("g must be a finite real number")
    if not np.isfinite(omega) or not np.isreal(omega) or omega <= 0:
        raise ValueError("omega must be a finite real number greater than zero")
    eta = shift[:, :, None] - lam
    H = np.zeros((K, L, K, L), dtype=np.float64)
    for A in range(K):
        for B in range(K):
            for p in range(L):
                for q in range(L):
                    s = np.exp(-np.sum((eta[A,:,p] - eta[B,:,q])**2)/2)
                    H[A,p,B,q] += s * tmat[p, q]
                    if p == q:
                        H[A,p,B,q] += omega * np.sum(eta[A,:,p]*eta[B,:,p]) * s + g * (eta[A,p,p] + eta[B,p,p]) * s
    return H


def lf_noci_lowest(
    hmat: np.ndarray,
    smat: np.ndarray,
    *,
    overlap_cut: float = 1e-10,
) -> tuple[float, np.ndarray, int]:
    """Solve a single- or multi-electron NOCI problem by orthogonalization.

    Kernels are flattened in C order only at this solver boundary. The
    basis address is A*L+p for four-index input, or
    (A*nstra+ia)*nstrb+ib for six-index input. Overlap eigenvectors are
    retained when their eigenvalue exceeds overlap_cut * max(1, s_max).
    The retained basis X satisfies X.T @ S @ X = I; the electronic and
    phonon model is entirely specified by the supplied H and S kernels.

    Parameters
    ----------
    hmat
        Finite symmetric float64 Hamiltonian kernel. Supported shapes are
        (K, L, K, L), with indices [A, p, B, q], and
        (K, nstra, nstrb, K, nstra, nstrb), with indices
        [A, ia, ib, B, ja, jb]. All dimensions must be positive.
    smat
        Finite symmetric float64 Gram kernel with exactly the same shape
        and ordering as hmat. Repeated frames may make it singular.
    overlap_cut
        Finite positive real scalar controlling retained overlap rank.
        Substantially negative overlap eigenvalues are rejected using
        the same threshold as in the existing single-electron solver.

    Returns
    -------
    energy
        Lowest variational energy in the retained span, as a Python float.
    coeff
        Float64 CI coefficients with shape (K, L) for four-index input or
        (K, nstra, nstrb) for six-index input. They obey
        coeff.ravel() @ S @ coeff.ravel() = 1. Their overall sign is
        arbitrary; individual squared entries are not probabilities in
        this nonorthogonal basis.
    rank
        Number of retained overlap eigenvectors, as a Python int.

    Raises
    ------
    ValueError
        If shapes, finite values, symmetry, cutoff, or the overlap
        spectrum violate the solver contract.
    TypeError
        If either kernel is not a NumPy array with float64 dtype.

    Notes
    -----
    No input arrays are modified. After overlap truncation, the relevant
    generalized residual is projected into the retained subspace. The
    full coordinate residual can be larger. This is a dense solver: its
    matrix dimension is K*L or K*nstra*nstrb. No lattice size, electron
    count, coupling convention, or model parameter is hard-coded here.
    """
    for name, array in (("hmat", hmat), ("smat", smat)):
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array")
        if array.ndim not in (4, 6):
            raise ValueError(f"{name} must be a four- or six-index kernel")
        half = array.ndim // 2
        if (any(size == 0 for size in array.shape)
                or array.shape[:half] != array.shape[half:]):
            raise ValueError(f"{name} must have matching nonempty bra and ket shapes")
        if array.dtype != np.float64:
            raise TypeError(f"{name} must have float64 dtype")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
    if smat.shape != hmat.shape:
        raise ValueError("smat must have the same shape as hmat")
    if (isinstance(overlap_cut, (bool, np.bool_))
            or not isinstance(overlap_cut, (int, float, np.integer, np.floating))
            or not np.isfinite(overlap_cut) or overlap_cut <= 0):
        raise ValueError("overlap_cut must be finite, real, and positive")

    basis_shape = hmat.shape[:hmat.ndim // 2]
    dimension = int(np.prod(basis_shape))
    hmat_flat = hmat.reshape(dimension, dimension)
    smat_flat = smat.reshape(dimension, dimension)
    if (not np.array_equal(hmat_flat, hmat_flat.T)
            or not np.array_equal(smat_flat, smat_flat.T)):
        raise ValueError("hmat and smat must be symmetric")

    s, vectors = np.linalg.eigh(smat_flat)
    threshold = max(float(s[-1]), 1.0) * overlap_cut
    if s[0] < -threshold:
        raise ValueError("smat has eigenvalues below the allowed overlap_cut threshold")
    keep = s > threshold
    if not np.any(keep):
        raise ValueError("No eigenvalues of smat are above the allowed overlap_cut threshold")
    X = vectors[:, keep] / np.sqrt(s[keep])
    energies, eigenvectors = np.linalg.eigh(X.T @ hmat_flat @ X)
    coeff = (X @ eigenvectors[:, 0]).reshape(basis_shape)
    return float(energies[0]), coeff, int(np.count_nonzero(keep))


def lf_translation_orbit(
    lam: np.ndarray,
    shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate the full cyclic orbit of one local LF reference frame.

    Translation ``R`` moves both the physical phonon mode and
    electron-site indices forward on an L-site ring:
    ``lam_orbit[R, x, p] = lam[(x-R) % L, (p-R) % L]`` and
    ``shift_orbit[R, x] = shift[(x-R) % L]``.  The orbit keeps
    all L positions, including duplicates of a symmetric seed.

    Parameters
    ----------
    lam
        Finite float64 LF parameters for one frame with shape
        ``(L, L)`` and index order ``[phonon mode, electron site]``.
    shift
        Finite float64 coherent shift for the same frame with
        shape ``(L,)`` and index order ``[phonon mode]``.

    Returns
    -------
    lam_orbit
        Translated LF parameters with shape ``(L, L, L)`` and
        index order ``[translation R, phonon mode x, electron site p]``.
    shift_orbit
        Translated coherent shifts with shape ``(L, L)`` and
        index order ``[translation R, phonon mode x]``.

    Raises
    ------
    ValueError
        If dimensions are empty or incompatible, or values are
        nonfinite.
    TypeError
        If either array does not have float64 dtype.

    Notes
    -----
    A full LF-HF orbital would also translate as
    ``coeff_R[p] = coeff[(p-R) % L]``.  The NOCI frame basis used
    here already includes every electron site, so electronic CI
    coefficients are optimized later and are not part of this output.
    """
    if lam.ndim != 2 or lam.shape[0] != lam.shape[1] or lam.shape[0] == 0:
        raise ValueError("lam must be a nonempty square 2D array")
    L = lam.shape[0]
    if shift.ndim != 1 or shift.shape != (L,):
        raise ValueError("shift must have shape (L,)")
    if lam.dtype != np.float64 or shift.dtype != np.float64:
        raise TypeError("lam and shift must have float64 dtype")
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(shift)):
        raise ValueError("lam and shift must contain only finite values")
    lam_orbit = np.zeros((L, L, L), dtype=np.float64)
    shift_orbit = np.zeros((L, L), dtype=np.float64)
    for R in range(L):
        lam_orbit[R, :, :] = np.roll(lam, shift=R, axis=(0, 1))
        shift_orbit[R, :] = np.roll(shift, shift=R, axis=0)
    return lam_orbit, shift_orbit


def lf_noci_site_density(
    coeff: np.ndarray,
    smat: np.ndarray,
) -> np.ndarray:
    """Return electronic site occupations for a real NOCI state.

    For ``|Psi> = sum_{A,p} coeff[A,p] |A,p>``, the occupation
    at site ``p`` is ``sum_{A,B} coeff[A,p] *
    smat[A,p,B,p] * coeff[B,p]``.  Cross-frame overlap terms
    are required even though different electronic sites are
    orthogonal.

    Parameters
    ----------
    coeff
        Finite float64 CI coefficients with shape ``(K, L)``
        and index order ``[frame, electron site]``.  Coefficients
        from ``lf_noci_lowest`` are normalized in the S metric.
    smat
        Finite float64 overlap tensor for the same frames with shape
        ``(K, L, K, L)`` and order ``[A, p, B, q]``.

    Returns
    -------
    numpy.ndarray
        Float64 site occupations with shape ``(L,)`` and order
        ``[electron site]``.  Their sum is
        ``coeff.ravel() @ S @ coeff.ravel()``, hence one for a
        normalized one-electron state.  The function neither
        renormalizes coefficients nor clips the result.

    Raises
    ------
    ValueError
        If dimensions are empty or incompatible, or values are
        nonfinite.
    TypeError
        If either array does not have float64 dtype.
    """
    if coeff.ndim != 2:
        raise ValueError("coeff must be a 2D array")
    K, L = coeff.shape
    if K == 0 or L == 0:
        raise ValueError("coeff must have positive dimensions")
    if smat.shape != (K, L, K, L):
        raise ValueError("smat must have shape (K, L, K, L)")
    if coeff.dtype != np.float64 or smat.dtype != np.float64:
        raise TypeError("coeff and smat must have float64 dtype")
    if not np.all(np.isfinite(coeff)) or not np.all(np.isfinite(smat)):
        raise ValueError("coeff and smat must contain only finite values")
    rho = np.zeros((L,), dtype=np.float64)
    for p in range(L):
        rho[p] = coeff[:,p] @ smat[:,p,:,p] @ coeff[:,p]
    return rho


def lf_noci_site_phonon_moment(
    coeff: np.ndarray,
    smat: np.ndarray,
    lam: np.ndarray,
    shift: np.ndarray,
) -> np.ndarray:
    """Return the joint electron-site and phonon-displacement moment.

    For real one-electron NOCI coefficients and conditional displacements
    ``eta[A, x, p] = shift[A, x] - lam[A, x, p]``, compute
    ``M[x, p] = <n_p (b_x + b_x^dagger)>``.  The matrix element sums over
    both frame indices: ``coeff[A, p] * smat[A, p, B, p] *
    coeff[B, p] * (eta[A, x, p] + eta[B, x, p])``.  In the uncentered
    Holstein convention, the coupling energy is ``g * trace(M)``.

    Parameters
    ----------
    coeff
        Finite float64 CI coefficients with shape ``(K, L)`` and index
        order ``[frame, electron site]``.  They are normally normalized
        in the metric supplied by ``smat``.
    smat
        Finite float64 overlap tensor with shape ``(K, L, K, L)`` and
        index order ``[A, p, B, q]`` for the same frames.
    lam
        Finite float64 LF parameters with shape ``(K, L, L)`` and index
        order ``[frame, phonon mode, electron site]``.
    shift
        Finite float64 coherent shifts with shape ``(K, L)`` and index
        order ``[frame, phonon mode]``.

    Returns
    -------
    numpy.ndarray
        Float64 joint moment with shape ``(L, L)`` and index order
        ``[phonon mode x, electron site p]``.  It is not divided by the
        site occupation, and inputs are not renormalized or modified.

    Raises
    ------
    ValueError
        If dimensions are empty or incompatible, or values are nonfinite.
    TypeError
        If an input array does not have float64 dtype.
    """
    if coeff.ndim != 2:
        raise ValueError("coeff must be a 2D array")
    K, L = coeff.shape
    if K == 0 or L == 0:
        raise ValueError("coeff must have positive dimensions")
    if smat.shape != (K, L, K, L):
        raise ValueError("smat must have shape (K, L, K, L)")
    if lam.shape != (K, L, L):
        raise ValueError("lam must have shape (K, L, L)")
    if shift.shape != (K, L):
        raise ValueError("shift must have shape (K, L)")
    if coeff.dtype != np.float64 or smat.dtype != np.float64 or lam.dtype != np.float64 or shift.dtype != np.float64:
        raise TypeError("coeff, smat, lam, and shift must have float64 dtype")
    if not np.all(np.isfinite(coeff)) or not np.all(np.isfinite(smat)) or not np.all(np.isfinite(lam)) or not np.all(np.isfinite(shift)):
        raise ValueError("coeff, smat, lam, and shift must contain only finite values")
    eta  = shift[:,:,None] - lam
    M = np.zeros((L,L), dtype=np.float64)
    for x in range(L):
        for p in range(L):
            for A in range(K):
                for B in range(K):
                    M[x,p] += coeff[A,p] * smat[A,p,B,p] * coeff[B,p] * (eta[A,x,p] + eta[B,x,p])
    return M


def lf_noci_trial_orbit(
    tmat: np.ndarray,
    g: float,
    omega: float,
    base_lam: np.ndarray,
    base_shift: np.ndarray,
    seed_lam: np.ndarray,
    seed_shift: np.ndarray,
    *,
    overlap_cut: float = 1e-10
) -> tuple[float, float, int, int]:
    """Measure the NOCI energy change from one full translation orbit.

    Solve the lowest state in the existing frame span, then append every
    cyclic translation of one candidate frame and rebuild both full kernels.
    The augmented kernels include cross terms between the original and new
    frames.  No input array is modified.

    Parameters
    ----------
    tmat
        Finite symmetric float64 site hopping matrix with shape ``(L, L)``
        and index order ``[p, q]``.
    g
        Finite real uncentered local electron-phonon coupling.
    omega
        Finite positive phonon frequency.
    base_lam
        Finite float64 LF parameters for ``K >= 1`` existing frames,
        with shape ``(K, L, L)`` and order ``[frame, mode, site]``.
    base_shift
        Finite float64 shifts with shape ``(K, L)`` and order
        ``[frame, mode]``.
    seed_lam
        Finite float64 LF parameters for one candidate frame, with
        shape ``(L, L)`` and order ``[mode, site]``.
    seed_shift
        Finite float64 shifts for the candidate with shape ``(L,)``.
    overlap_cut
        Positive relative overlap-eigenvalue threshold passed to
        ``lf_noci_lowest`` for both solves.

    Returns
    -------
    energy_base
        Lowest NOCI energy in the existing frame span.
    energy_augmented
        Lowest NOCI energy after adding the candidate's full orbit.
    rank_base
        Retained overlap rank of the existing span.
    rank_augmented
        Retained overlap rank after augmentation.

    Raises
    ------
    ValueError
        If dimensions, finite values, symmetry, or solver parameters are invalid.
    TypeError
        If an input array does not have float64 dtype.

    Notes
    -----
    The variational energy gain is ``energy_base - energy_augmented``.
    Duplicate or physically equivalent frames need not increase rank.
    Numerical overlap truncation can break exact subspace nesting, so this
    function does not clip a small negative computed gain.
    """
    H = lf_frame_hamiltonian(tmat, g, omega, base_lam, base_shift)
    S = lf_frame_overlap(base_lam, base_shift)
    Energy, _, rank = lf_noci_lowest(H, S, overlap_cut=overlap_cut)
    lam_orbit, shift_orbit = lf_translation_orbit(seed_lam, seed_shift)
    if lam_orbit.shape[1:] != base_lam.shape[1:]:
        raise ValueError("candidate frame size must match the base frames")
    lam = np.append(base_lam, lam_orbit, axis=0)
    shift = np.append(base_shift, shift_orbit, axis=0)
    H_new = lf_frame_hamiltonian(tmat, g, omega, lam, shift)
    S_new = lf_frame_overlap(lam, shift)
    Energy_new, _, rank_new = lf_noci_lowest(H_new, S_new, overlap_cut=overlap_cut)
    return Energy, Energy_new, rank, rank_new


def lf_noci_score_orbits(
    tmat: np.ndarray,
    g: float,
    omega: float,
    base_lam: np.ndarray,
    base_shift: np.ndarray,
    candidate_lam: np.ndarray,
    candidate_shift: np.ndarray,
    *,
    overlap_cut: float = 1e-10,
) -> tuple[float, int, np.ndarray, np.ndarray]:
    """Score each candidate LF orbit against the same fixed NOCI base.

    Candidate ``j`` is evaluated by adding only its full translation
    orbit to the existing frames and solving the lowest generalized
    eigenstate.  Candidates are not appended cumulatively, sorted,
    filtered, or selected.

    Parameters
    ----------
    tmat
        Finite symmetric float64 hopping matrix with shape ``(L, L)``
        and index order ``[p, q]``.
    g
        Finite real uncentered local electron-phonon coupling.
    omega
        Finite positive phonon frequency.
    base_lam
        Finite float64 parameters of ``K >= 1`` base frames, with shape
        ``(K, L, L)`` and order ``[frame, mode, site]``.
    base_shift
        Finite float64 base shifts with shape ``(K, L)``.
    candidate_lam
        Finite float64 candidate seeds with shape ``(J, L, L)``,
        ``J >= 1``, and order ``[candidate, mode, site]``.
    candidate_shift
        Finite float64 candidate shifts with shape ``(J, L)``.
    overlap_cut
        Positive relative overlap cutoff used for every trial solve.

    Returns
    -------
    energy_base
        Lowest NOCI energy in the fixed base span.
    rank_base
        Retained overlap rank of the base span.
    energy_trial
        Float64 array with shape ``(J,)``: lowest energies after
        adding each candidate orbit separately, in input order.
    rank_trial
        Int64 array with shape ``(J,)``: corresponding retained ranks.

    Raises
    ------
    ValueError
        If shapes are empty or incompatible, values are nonfinite, or
        model and solver parameters are invalid.
    TypeError
        If an input array does not have float64 dtype.

    Notes
    -----
    The energy gain of candidate ``j`` is
    ``energy_base - energy_trial[j]``.  The result is not the energy
    of a space containing all candidates at once.  Input arrays are
    neither modified nor renormalized.
    """
    if base_lam.ndim != 3 or base_lam.shape[0] == 0 or base_lam.shape[1] == 0 or base_lam.shape[1] != base_lam.shape[2]:
        raise ValueError("base_lam must have nonempty shape (K, L, L)")
    K, L, _ = base_lam.shape
    if base_shift.shape != (K, L):
        raise ValueError("base_shift must have shape (K, L)")
    if candidate_lam.ndim != 3 or candidate_lam.shape[0] == 0 or candidate_lam.shape[1:] != (L, L):
        raise ValueError("candidate_lam must have nonempty shape (J, L, L)")
    J = candidate_lam.shape[0]
    if candidate_shift.shape != (J, L):
        raise ValueError("candidate_shift must have shape (J, L)")
    arrays = (base_lam, base_shift, candidate_lam, candidate_shift)
    if any(array.dtype != np.float64 for array in arrays):
        raise TypeError("LF frame arrays must have float64 dtype")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("LF frame arrays must contain only finite values")
    candidate_energy = np.zeros((J,), dtype=np.float64)
    candidate_rank = np.zeros((J,), dtype=np.int64)
    base_energy = 0.0
    base_rank = 0
    for j in range(J):
        if j == 0:
            base_energy, trial_energy, base_rank, trial_rank = lf_noci_trial_orbit(
                tmat,
                g,
                omega,
                base_lam,
                base_shift,
                candidate_lam[j],
                candidate_shift[j],
                overlap_cut=overlap_cut,
            )
        else:
            _, trial_energy, _, trial_rank = lf_noci_trial_orbit(
                tmat,
                g,
                omega,
                base_lam,
                base_shift,
                candidate_lam[j],
                candidate_shift[j],
                overlap_cut=overlap_cut,
            )
        candidate_energy[j] = trial_energy
        candidate_rank[j] = trial_rank
    return base_energy, base_rank, candidate_energy, candidate_rank


def _validate_multi_frame_inputs(
    lam: np.ndarray,
    shift: np.ndarray,
    nelec: tuple[int, int],
) -> tuple[int, int]:
    """Validate multi-electron LF frames and return ``(nframe, nsite)``.

    ``lam[A, x, p]`` has frame, phonon-mode, and electron-site indices;
    ``shift[A, x]`` has frame and phonon-mode indices. Both are finite
    float64 arrays. ``nelec`` gives fixed alpha and beta electron counts.
    This helper performs no model contraction and does not alter inputs.
    """
    if (not isinstance(lam, np.ndarray) or lam.ndim != 3
            or lam.shape[0] == 0 or lam.shape[1] == 0
            or lam.shape[1] != lam.shape[2]):
        raise ValueError("lam must have nonempty shape (K, L, L)")
    nframe, nsite, _ = lam.shape
    if not isinstance(shift, np.ndarray) or shift.shape != (nframe, nsite):
        raise ValueError("shift must have shape (K, L)")
    if lam.dtype != np.float64 or shift.dtype != np.float64:
        raise TypeError("lam and shift must have float64 dtype")
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(shift)):
        raise ValueError("lam and shift must contain only finite values")
    if (not isinstance(nelec, tuple) or len(nelec) != 2
            or any(isinstance(n, (bool, np.bool_))
                   or not isinstance(n, (int, np.integer))
                   or n < 0 or n > nsite for n in nelec)):
        raise ValueError("nelec must be (neleca, nelecb), each between 0 and L")
    return nframe, nsite


def lf_multi_frame_overlap(
    lam: np.ndarray,
    shift: np.ndarray,
    nelec: tuple[int, int],
) -> np.ndarray:
    """Return the site-determinant Gram tensor of multi-electron LF frames.

    A frame consists of an orthonormal site determinant and its conditional
    multimode coherent state. The local coupling is uncentered. The alpha
    and beta determinant addresses follow ``make_electron_basis`` order.

    Parameters
    ----------
    lam
        Full LF matrices with shape ``(K, L, L)`` and index order
        ``[frame A, phonon mode x, electron site p]``; finite float64.
    shift
        Coherent shifts with shape ``(K, L)`` and order ``[A, x]``;
        finite float64.
    nelec
        Fixed electron counts ``(neleca, nelecb)`` with each count from
        zero through L.

    Returns
    -------
    numpy.ndarray
        Real Gram tensor with shape ``(K, nstra, nstrb, K, nstra, nstrb)``
        and index order ``[A, ia, ib, B, ja, jb]``, where
        ``nstr_sigma = comb(L, nelec_sigma)``. The tensor retains separate
        alpha and beta determinant axes. Its diagonal frame blocks are
        identity matrices on the site determinant space.

    Notes
    -----
    This is the multi-electron extension of ``lf_frame_overlap``. It is
    independent of the phonon Fock cutoff. The corresponding Hamiltonian
    kernel will use the same conditional coherent displacements.
    """
    nframe, nsite = _validate_multi_frame_inputs(lam, shift, nelec)
    str_a, _ = my_direct_ep.make_electron_basis(nsite, nelec[0])
    str_b, _ = my_direct_ep.make_electron_basis(nsite, nelec[1])

    sites = np.arange(nsite, dtype=np.int64)

    occ_a = ((str_a[:, None] >> sites[None, :] & 1).astype(np.float64))
    occ_b = ((str_b[:, None] >> sites[None, :] & 1).astype(np.float64))
    occupation = occ_a[:, None, :] + occ_b[None, :, :]

    eta = shift[:, None, None, :] - np.einsum("Axp,ijp->Aijx", lam, occupation)

    difference = eta[:, None, :, :, :] - eta[None, :, :, :, :]
    same_det_overlap = np.exp(-0.5 * np.sum(difference**2, axis=-1))
    identity_a = np.eye(len(str_a), dtype=np.float64)
    identity_b = np.eye(len(str_b), dtype=np.float64)

    smat = np.einsum("ABij,ik,jl->AijBkl", same_det_overlap, identity_a, identity_b)
    return smat


def lf_multi_frame_hamiltonian(
    tmat: np.ndarray,
    U: float,
    g: float,
    omega: float,
    lam: np.ndarray,
    shift: np.ndarray,
    nelec: tuple[int, int],
) -> np.ndarray:
    """Return the physical Hamiltonian kernel between multi-electron LF frames.

    Uses the uncentered coupling g*n[x]*(b[x] + b[x].dagger), local Hubbard
    interaction U*n_alpha[x]*n_beta[x], and omega*b[x].dagger*b[x] without
    a zero-point constant. The result already uses the paper energy
    convention; no centered-to-paper energy correction is added.

    Parameters
    ----------
    tmat
        Finite float64 hopping matrix with shape (L, L), indexed by
        electron sites [p, q]. Must be exactly symmetric; diagonal entries
        are allowed. No particular lattice or boundary condition is assumed.
    U, g, omega
        Finite real interaction, electron-phonon coupling, and positive
        common phonon frequency.
    lam
        Finite float64 LF matrices with shape (K, L, L) and index order
        [frame A, phonon mode x, electron site p].
    shift
        Finite float64 coherent shifts with shape (K, L), ordered [A, x].
    nelec
        Fixed electron counts (neleca, nelecb), each between zero and L.

    Returns
    -------
    numpy.ndarray
        Float64 tensor of shape (K, nstra, nstrb, K, nstra, nstrb), ordered
        [A, ia, ib, B, ja, jb], matching lf_multi_frame_overlap. Electronic
        determinant addresses use my_direct_ep.make_electron_basis order.

    Notes
    -----
    Each basis ket is a site determinant times its conditional normalized
    coherent state. Coherent-state matrix elements are evaluated in the
    complete phonon space, so there is no Nmax parameter. Hopping changes
    the electron determinant: its coherent overlap must be retained even
    when the full state overlap from lf_multi_frame_overlap is zero.
    The resulting frame space is variational and need not span the exact
    electron-phonon ground state.
    """
    nframe, nsite = _validate_multi_frame_inputs(lam, shift, nelec)
    if not isinstance(tmat, np.ndarray) or tmat.shape != (nsite, nsite):
        raise ValueError("tmat must have shape (L, L)")
    if tmat.dtype != np.float64:
        raise TypeError("tmat must have float64 dtype")
    if not np.all(np.isfinite(tmat)):
        raise ValueError("tmat must contain only finite values")
    if not np.array_equal(tmat, tmat.T):
        raise ValueError("tmat must be symmetric")
    for name, value in (("U", U), ("g", g), ("omega", omega)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(value)):
            raise ValueError(f"{name} must be a finite real scalar")
    if omega <= 0:
        raise ValueError("omega must be positive")
    str_a, link_a = my_direct_ep.make_electron_basis(nsite, nelec[0])
    str_b, link_b = my_direct_ep.make_electron_basis(nsite, nelec[1])

    sites = np.arange(nsite, dtype=np.int64)

    occ_a = ((str_a[:, None] >> sites[None, :] & 1).astype(np.float64))
    occ_b = ((str_b[:, None] >> sites[None, :] & 1).astype(np.float64))
    occupation = occ_a[:, None, :] + occ_b[None, :, :]

    eta = shift[:, None, None, :] - np.einsum("Axp,ijp->Aijx", lam, occupation)

    double_occ = occ_a @ occ_b.T

    create, annihilate, target, sign = link_a.transpose(2, 0, 1)
    source = np.broadcast_to(np.arange(len(str_a))[:, None], target.shape)
    hopping_a = np.zeros((len(str_a), len(str_a)))
    np.add.at(hopping_a, (target, source), sign * tmat[create, annihilate])

    create, annihilate, target, sign = link_b.transpose(2, 0, 1)
    source = np.broadcast_to(np.arange(len(str_b))[:, None], target.shape)
    hopping_b = np.zeros((len(str_b), len(str_b)))
    np.add.at(hopping_b, (target, source), sign * tmat[create, annihilate])

    identity_a = np.eye(len(str_a))
    identity_b = np.eye(len(str_b))
    hopping = (
        np.einsum("ik,jl->ijkl", hopping_a, identity_b)
        + np.einsum("ik,jl->ijkl", identity_a, hopping_b)
    )

    eta_dot = np.einsum("Aijx,Bklx->AijBkl", eta, eta)
    eta_norm2 = np.sum(eta**2, axis=-1)

    distance2 = (
        eta_norm2[:, :, :, None, None, None]
        + eta_norm2[None, None, None, :, :, :]
        - 2 * eta_dot
    )
    G = np.exp(-0.5 * distance2)
    H_hop = hopping[None, :, :, None, :, :] * G
    same_eta_dot = np.einsum("Aijx,Bijx->ABij", eta, eta)
    density_eta = np.einsum("ijx,Aijx->Aij", occupation, eta)
    V = (
        U * double_occ
        + omega * same_eta_dot
        + g * (density_eta[:, None, :, :] + density_eta[None, :, :, :])
    )
    diagonal_tensor = np.einsum("ABij,ik,jl->AijBkl", V, identity_a, identity_b)
    smat = lf_multi_frame_overlap(lam, shift, nelec)
    return H_hop + diagonal_tensor * smat
