"""Acceptance tests for one fixed-reference LF-HF/LF-MP2 point."""

import math

import numpy as np
import pytest

from src.lf_mp import lf_effective_one_body, lf_mp2_reference_point
from src.my_direct_ep import electron_ring_hopping

EXPECTED_KEYS = {
    "hf_energy",
    "mp2_correction",
    "total_energy",
    "max_total",
    "coeff",
    "mo_energy",
    "mo_coeff",
    "occupations",
    "total_occupations",
    "pure_matrix_elements",
    "single_matrix_elements",
    "pure_denominators",
    "single_denominators",
}


def test_two_site_zero_displacement_matches_hand_calculated_channels():
    tmat = np.array([[0.0, -1.0], [-1.0, 0.0]])
    g = 0.4
    omega = 0.7

    result = lf_mp2_reference_point(
        tmat,
        g,
        omega,
        shift=np.zeros(2),
        lam=np.zeros((2, 2)),
        max_total=1,
    )

    expected_pure = -(g**2) / (2.0 * omega)
    expected_single = -(g**2) / (2.0 * (2.0 + omega))
    expected_mp2 = expected_pure + expected_single
    assert set(result) == EXPECTED_KEYS
    assert result["hf_energy"] == pytest.approx(-1.0, abs=1e-14)
    assert result["mp2_correction"] == pytest.approx(expected_mp2, abs=1e-14)
    assert result["total_energy"] == pytest.approx(-1.0 + expected_mp2, abs=1e-14)
    assert result["max_total"] == 1
    np.testing.assert_allclose(result["mo_energy"], [-1.0, 1.0], atol=1e-14)
    np.testing.assert_allclose(
        np.abs(result["pure_matrix_elements"]),
        np.full(2, g / 2.0),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.abs(result["single_matrix_elements"][:, 0, 0]),
        np.full(2, g / 2.0),
        atol=1e-14,
    )
    np.testing.assert_allclose(result["pure_denominators"], -omega, atol=1e-14)
    np.testing.assert_allclose(
        result["single_denominators"],
        -(2.0 + omega),
        atol=1e-14,
    )


def test_four_site_zero_coupling_has_zero_correction_and_auditable_shapes():
    norb = 4
    max_total = 3
    nconfig = math.comb(max_total + norb, norb) - 1
    tmat = electron_ring_hopping(norb, -1.0)

    result = lf_mp2_reference_point(
        tmat,
        g=0.0,
        omega=0.5,
        shift=np.zeros(norb),
        lam=np.zeros((norb, norb)),
        max_total=max_total,
    )

    assert result["hf_energy"] == pytest.approx(-2.0, abs=1e-14)
    assert result["mp2_correction"] == 0.0
    assert result["total_energy"] == pytest.approx(-2.0, abs=1e-14)
    assert result["coeff"].shape == (norb,)
    assert result["mo_energy"].shape == (norb,)
    assert result["mo_coeff"].shape == (norb, norb)
    assert result["occupations"].shape == (nconfig, norb)
    assert result["total_occupations"].shape == (nconfig,)
    assert result["pure_matrix_elements"].shape == (nconfig,)
    assert result["pure_denominators"].shape == (nconfig,)
    assert result["single_matrix_elements"].shape == (nconfig, 1, norb - 1)
    assert result["single_denominators"].shape == (nconfig, 1, norb - 1)
    np.testing.assert_array_equal(
        result["total_occupations"],
        np.sum(result["occupations"], axis=1),
    )
    heff = lf_effective_one_body(
        tmat,
        g=0.0,
        omega=0.5,
        shift=np.zeros(norb),
        lam=np.zeros((norb, norb)),
    )
    np.testing.assert_allclose(
        heff @ result["mo_coeff"],
        result["mo_coeff"] * result["mo_energy"][None, :],
        atol=1e-14,
    )
    np.testing.assert_allclose(result["coeff"], result["mo_coeff"][:, 0], atol=0.0)


def test_simultaneous_row_shift_is_a_gauge_transformation():
    tmat = np.array([[0.0, -1.0], [-1.0, 0.0]])
    g = 0.4
    omega = 0.7
    lam = np.array([[0.2, -0.1], [0.05, 0.3]])
    shift = np.zeros(2)
    gauge_offset = np.array([0.17, -0.23])

    gauge_fixed = lf_mp2_reference_point(
        tmat, g, omega, shift, lam, max_total=3
    )
    shifted = lf_mp2_reference_point(
        tmat,
        g,
        omega,
        shift + gauge_offset,
        lam + gauge_offset[:, None],
        max_total=3,
    )

    assert shifted["hf_energy"] == pytest.approx(gauge_fixed["hf_energy"], abs=1e-14)
    assert shifted["mp2_correction"] == pytest.approx(
        gauge_fixed["mp2_correction"], abs=1e-14
    )
    assert shifted["total_energy"] == pytest.approx(
        gauge_fixed["total_energy"], abs=1e-14
    )
    np.testing.assert_allclose(
        shifted["mo_energy"] - shifted["mo_energy"][0],
        gauge_fixed["mo_energy"] - gauge_fixed["mo_energy"][0],
        atol=1e-14,
    )
