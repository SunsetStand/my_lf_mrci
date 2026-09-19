"""Lang--Firsov mean-field reference and perturbative corrections."""

import numpy as np
from scipy import optimize as scipy_optimize

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
