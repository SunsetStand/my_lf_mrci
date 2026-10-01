"""Independent electronic FCI checks for the scan's energies and densities."""
import numpy as np
import pytest
from pyscf import fci, lib

from scripts import fig2d_mr_scan as scan


@pytest.mark.parametrize('displaced', [False, True])
def test_uniform_electronic_frame_matches_independent_fci(displaced):
    lib.num_threads(1)
    tmat = scan.ep.electron_ring_hopping(4, -1.0)
    g, omega = (0.7, 0.5) if displaced else (0.0, 0.5)
    z = np.array([-0.1, -0.6, 0.2, -0.4]) if displaced else np.zeros(4)
    # Identical phonon states in two redundant frames; independent Hubbard FCI.
    lam = np.zeros((2, 4, 4), dtype=np.float64)
    shift = np.tile(z, (2, 1))
    strings, _ = scan.ep.make_electron_basis(4, 2)
    occ = ((strings[:, None] >> np.arange(4)) & 1).astype(np.float64)
    result, _, density = scan.measure(tmat, g, omega, lam, shift, occ)
    eri = np.zeros((4, 4, 4, 4))
    for site in range(4):
        eri[site, site, site, site] = scan.U
    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-13
    energy, ci = solver.kernel(tmat + np.diag(2*g*z), eri, 4, (2, 2))
    assert solver.converged
    rdms = solver.make_rdm1s(ci, 4, (2, 2))
    assert result['energy'] == pytest.approx(energy + omega*(z@z), abs=1e-10)
    np.testing.assert_allclose(density, np.array([np.diag(dm) for dm in rdms]), atol=1e-10)
    assert result['rank'] == 36
    assert result['residual_projected'] < 1e-10
