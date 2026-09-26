import numpy as np


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
    """Solve the lowest NOCI state after canonical overlap orthogonalization.

    The four-index kernels are flattened only at this solver boundary
    using ``a = A*L + p``.  Eigenvectors of the overlap below
    ``overlap_cut * max(1, s_max)`` are discarded.  In the retained
    subspace, the function diagonalizes ``X.T @ H @ X`` with
    ``X.T @ S @ X = I``.  It does not alter the input kernels.

    Parameters
    ----------
    hmat
        Finite symmetric float64 Hamiltonian kernel with shape
        ``(K, L, K, L)`` and index order ``[A, p, B, q]``.
    smat
        Finite symmetric float64 Gram kernel with the same shape and
        index order.  It may be singular when frames are repeated.
    overlap_cut
        Finite positive relative cutoff.  An overlap eigenvalue is
        retained only if it exceeds
        ``overlap_cut * max(1, s_max)``.

    Returns
    -------
    energy
        Lowest variational energy in the retained span.
    coeff
        Real CI coefficients with shape ``(K, L)``, ordered as
        ``[frame, electron site]`` and normalized by
        ``coeff.ravel() @ S @ coeff.ravel() = 1``.  Their overall
        sign is arbitrary; individual squared entries are not
        probabilities in this nonorthogonal basis.
    rank
        Number of overlap eigenvectors retained after cutoff.

    Raises
    ------
    ValueError
        If dimensions, finite values, symmetry, cutoff, or the Gram
        matrix spectrum violate the solver contract.
    TypeError
        If either kernel does not have float64 dtype.

    Notes
    -----
    When near-dependent directions are discarded, the generalized
    residual projected into the retained span is the relevant solver
    check.  The full coordinate residual may be larger.
    """
    if hmat.ndim != 4 or hmat.shape[0] != hmat.shape[2] or hmat.shape[1] != hmat.shape[3]:
        raise ValueError("hmat must be a 4D array with shape (K, L, K, L)")
    if smat.ndim != 4 or smat.shape[0] != smat.shape[2] or smat.shape[1] != smat.shape[3]:
        raise ValueError("smat must be a 4D array with shape (K, L, K, L)")
    K, L, _, _ = hmat.shape
    if K <= 0 or L <= 0:
        raise ValueError("hmat must have positive dimensions")
    if smat.shape != (K, L, K, L):
        raise ValueError("smat must have the same shape as hmat")
    if hmat.dtype != np.float64 or smat.dtype != np.float64:
        raise TypeError("hmat and smat must have float64 dtype")
    if not np.all(np.isfinite(hmat)) or not np.all(np.isfinite(smat)):
        raise ValueError("hmat and smat must contain only finite values")
    if not np.isfinite(overlap_cut) or not np.isreal(overlap_cut) or overlap_cut <= 0:
        raise ValueError("overlap_cut must be finite, real, and positive")
    hmat_flat = hmat.reshape(K*L, K*L)
    smat_flat = smat.reshape(K*L, K*L)
    if not np.array_equal(hmat_flat, hmat_flat.T) or not np.array_equal(smat_flat, smat_flat.T):
        raise ValueError("hmat and smat must be symmetric")
    s, V = np.linalg.eigh(smat_flat)
    tau = max(np.max(s), 1) * overlap_cut
    if np.min(s) < -tau:
        raise ValueError("smat has eigenvalues below the allowed overlap_cut threshold")
    s_keep = s > tau
    if not np.any(s_keep):
        raise ValueError("No eigenvalues of smat are above the allowed overlap_cut threshold")
    X = np.zeros((K*L, np.sum(s_keep)), dtype=np.float64)
    X[:,:] = V[:, s_keep] / np.sqrt(s[s_keep])
    energy, y = np.linalg.eigh(X.T @ hmat_flat @ X)
    C = X @ y
    rank = C.shape[1]
    return energy[0], C[:, 0].reshape(K, L), rank


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
