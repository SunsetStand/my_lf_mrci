"""Acceptance checks for electron density in a nonorthogonal LF frame basis."""

import numpy as np
import pytest

from src import lf_mr
from src.cs_mp import cs_site_density
from src.my_direct_ep import electron_ring_hopping


@pytest.fixture
def site_density():
    if not hasattr(lf_mr, "lf_noci_site_density"):
        pytest.skip("Student task pending: implement lf_noci_site_density")
    return lf_mr.lf_noci_site_density


def test_one_frame_reduces_to_ordinary_site_density(site_density):
    coeff = np.full((1, 4), 0.5)
    smat = lf_mr.lf_frame_overlap(np.zeros((1, 4, 4)),
                                 np.zeros((1, 4)))
    old_coeff, old_smat = coeff.copy(), smat.copy()
    rho = site_density(coeff, smat)
    assert rho.shape == (4,)
    assert rho.dtype == np.float64
    np.testing.assert_allclose(rho, cs_site_density(coeff[0]),
                               atol=1e-14, rtol=0)
    np.testing.assert_array_equal(coeff, old_coeff)
    np.testing.assert_array_equal(smat, old_smat)


def test_duplicate_frame_cross_term_is_essential(site_density):
    coeff = np.array([[0.5], [0.5]])
    smat = lf_mr.lf_frame_overlap(np.zeros((2, 1, 1)),
                                 np.zeros((2, 1)))
    rho = site_density(coeff, smat)
    np.testing.assert_allclose(rho, [1.0], atol=1e-14, rtol=0)
    assert np.sum(coeff[:, 0]**2) == 0.5


def test_two_frame_analytic_interference_with_signed_coefficients(site_density):
    lam = np.zeros((2, 2, 2))
    lam[1, 0, 0] = 0.8
    lam[1, 1, 1] = 1.2
    smat = lf_mr.lf_frame_overlap(lam, np.zeros((2, 2)))
    raw = np.array([[0.4, 0.2], [0.3, -0.1]])
    overlap0 = np.exp(-0.5 * 0.8**2)
    overlap1 = np.exp(-0.5 * 1.2**2)
    numerators = np.array([
        0.4**2 + 0.3**2 + 2 * 0.4 * 0.3 * overlap0,
        0.2**2 + (-0.1)**2 + 2 * 0.2 * (-0.1) * overlap1,
    ])
    coeff = raw / np.sqrt(numerators.sum())
    expected = numerators / numerators.sum()
    rho = site_density(coeff, smat)
    np.testing.assert_allclose(rho, expected, atol=1e-14, rtol=0)
    np.testing.assert_allclose(site_density(-coeff, smat), rho,
                               atol=1e-14, rtol=0)
    np.testing.assert_allclose(rho.sum(), 1.0, atol=1e-14, rtol=0)
    assert np.all(rho >= 0)


def test_translated_four_site_noci_density_is_uniform(site_density):
    rng = np.random.default_rng(906)
    lam = rng.normal(scale=0.4, size=(4, 4))
    shift = rng.normal(scale=0.2, size=4)
    lam_orbit, shift_orbit = lf_mr.lf_translation_orbit(lam, shift)
    smat = lf_mr.lf_frame_overlap(lam_orbit, shift_orbit)
    hmat = lf_mr.lf_frame_hamiltonian(
        electron_ring_hopping(4, -1.0), 0.65, 0.5,
        lam_orbit, shift_orbit,
    )
    _, coeff, rank = lf_mr.lf_noci_lowest(hmat, smat)
    assert rank == 16
    rho = site_density(coeff, smat)
    np.testing.assert_allclose(rho, np.full(4, 0.25),
                               atol=1e-10, rtol=0)
    np.testing.assert_allclose(rho.sum(), 1.0,
                               atol=1e-10, rtol=0)
