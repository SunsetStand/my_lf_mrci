"""Acceptance tests for one complete one-electron CS-MP2 point."""

import numpy as np
import pytest

from src.cs_mp import cs_fock, cs_mp2_point
from src.my_direct_ep import electron_ring_hopping

EXPECTED_KEYS = {
    "hf_energy",
    "mp2_correction",
    "total_energy",
    "coeff",
    "shift",
    "mo_energy",
    "mo_coeff",
    "niter",
    "density_residual",
}


@pytest.mark.parametrize("alpha", [0.0, 0.8])
def test_uniform_four_site_point_matches_analytic_cs_mp2_line(alpha):
    omega = 0.5
    coupling = np.sqrt(alpha * omega)
    hopping = electron_ring_hopping(4, -1.0)
    coeff0 = np.full(4, 0.5)

    result = cs_mp2_point(hopping, coupling, omega, coeff0)

    assert set(result) == EXPECTED_KEYS
    assert result["hf_energy"] == pytest.approx(-2.0 - alpha / 4.0, abs=1e-14)
    assert result["mp2_correction"] == pytest.approx(
        -23.0 * alpha / 180.0,
        abs=1e-14,
    )
    assert result["total_energy"] == pytest.approx(
        result["hf_energy"] + result["mp2_correction"],
        abs=1e-14,
    )
    assert result["total_energy"] == pytest.approx(
        -2.0 - 17.0 * alpha / 45.0,
        abs=1e-14,
    )
    assert result["coeff"].shape == (4,)
    assert result["shift"].shape == (4,)
    assert result["mo_energy"].shape == (4,)
    assert result["mo_coeff"].shape == (4, 4)
    assert result["niter"] == 1
    assert result["density_residual"] < 1e-10
    np.testing.assert_allclose(np.abs(result["coeff"]) ** 2, 0.25, atol=1e-14)
    np.testing.assert_allclose(
        result["mo_energy"],
        np.array([-2.0, 0.0, 0.0, 2.0]) - alpha / 2.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result["mo_coeff"].conj().T @ result["mo_coeff"],
        np.eye(4),
        atol=1e-14,
    )
    fock = cs_fock(hopping, coupling, result["shift"])
    np.testing.assert_allclose(
        fock @ result["mo_coeff"],
        result["mo_coeff"] * result["mo_energy"][None, :],
        atol=1e-14,
    )


def test_point_rejects_unconverged_cs_hf_reference():
    hopping = electron_ring_hopping(4, -1.0)

    with pytest.raises(RuntimeError, match="CS-HF"):
        cs_mp2_point(
            hopping,
            coupling=0.2,
            omega=0.5,
            coeff0=np.full(4, 0.5),
            max_cycle=0,
        )
