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
