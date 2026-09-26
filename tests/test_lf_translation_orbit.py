"""Acceptance checks for translation of a local one-electron LF frame."""

import numpy as np
import pytest

from src import lf_mr
from src.lf_mp import lf_hf_full_state
from src.my_direct_ep import electron_ring_hopping


@pytest.fixture
def orbit():
    if not hasattr(lf_mr, "lf_translation_orbit"):
        pytest.skip("Student task pending: implement lf_translation_orbit")
    return lf_mr.lf_translation_orbit


def test_one_localized_displacement_moves_both_indices(orbit):
    lam = np.zeros((4, 4))
    lam[0, 0] = 0.7
    shift = np.zeros(4)
    shift[0] = 0.2
    old_lam, old_shift = lam.copy(), shift.copy()
    lam_orbit, shift_orbit = orbit(lam, shift)
    assert lam_orbit.shape == (4, 4, 4)
    assert shift_orbit.shape == (4, 4)
    assert lam_orbit.dtype == shift_orbit.dtype == np.float64
    for r in range(4):
        expected_lam = np.zeros((4, 4))
        expected_lam[r, r] = 0.7
        expected_shift = np.zeros(4)
        expected_shift[r] = 0.2
        np.testing.assert_array_equal(lam_orbit[r], expected_lam)
        np.testing.assert_array_equal(shift_orbit[r], expected_shift)
    np.testing.assert_array_equal(lam, old_lam)
    np.testing.assert_array_equal(shift, old_shift)


def test_orbit_has_cyclic_index_order_for_asymmetric_seed(orbit):
    lam = np.arange(16, dtype=np.float64).reshape(4, 4) / 10
    shift = np.array([0.3, -0.2, 0.1, 0.4])
    lam_orbit, shift_orbit = orbit(lam, shift)
    for r in range(4):
        for x in range(4):
            assert shift_orbit[r, x] == shift[(x - r) % 4]
            for p in range(4):
                assert lam_orbit[r, x, p] == lam[(x - r) % 4, (p - r) % 4]


def test_physical_displacements_are_gauge_invariant(orbit):
    rng = np.random.default_rng(904)
    lam = rng.normal(size=(4, 4))
    shift = rng.normal(size=4)
    gauge = rng.normal(size=4)
    a_lam, a_shift = orbit(lam, shift)
    b_lam, b_shift = orbit(lam + gauge[:, None], shift + gauge)
    eta_a = a_shift[:, :, None] - a_lam
    eta_b = b_shift[:, :, None] - b_lam
    np.testing.assert_allclose(eta_a, eta_b, atol=1e-15, rtol=0)


def test_h_and_s_are_covariant_under_joint_translation(orbit):
    rng = np.random.default_rng(905)
    lam = rng.normal(scale=0.35, size=(4, 4))
    shift = rng.normal(scale=0.2, size=4)
    lam_orbit, shift_orbit = orbit(lam, shift)
    tmat = electron_ring_hopping(4, -1.0)
    hmat = lf_mr.lf_frame_hamiltonian(
        tmat, 0.55, 0.5, lam_orbit, shift_orbit
    )
    smat = lf_mr.lf_frame_overlap(lam_orbit, shift_orbit)
    for r in range(1, 4):
        np.testing.assert_allclose(
            hmat, np.roll(hmat, r, axis=(0, 1, 2, 3)),
            atol=1e-14, rtol=0,
        )
        np.testing.assert_allclose(
            smat, np.roll(smat, r, axis=(0, 1, 2, 3)),
            atol=1e-14, rtol=0,
        )


def test_orbit_noci_energy_does_not_exceed_seed_lf_hf(orbit):
    rng = np.random.default_rng(906)
    lam = rng.normal(scale=0.4, size=(4, 4))
    shift = rng.normal(scale=0.2, size=4)
    tmat = electron_ring_hopping(4, -1.0)
    g, omega = 0.65, 0.5
    seed_energy, _, _ = lf_hf_full_state(lam, shift, tmat, g, omega)
    lam_orbit, shift_orbit = orbit(lam, shift)
    hmat = lf_mr.lf_frame_hamiltonian(
        tmat, g, omega, lam_orbit, shift_orbit
    )
    smat = lf_mr.lf_frame_overlap(lam_orbit, shift_orbit)
    energy, coeff, rank = lf_mr.lf_noci_lowest(hmat, smat)
    assert 1 <= rank <= 16
    assert coeff.shape == (4, 4)
    assert energy <= seed_energy + 1e-10
    c = coeff.ravel()
    np.testing.assert_allclose(
        c @ smat.reshape(16, 16) @ c, 1.0, atol=1e-10, rtol=0
    )


def test_uniform_seed_is_kept_then_solver_removes_duplicates(orbit):
    lam_orbit, shift_orbit = orbit(np.zeros((4, 4)), np.zeros(4))
    assert lam_orbit.shape[0] == 4
    tmat = electron_ring_hopping(4, -1.0)
    hmat = lf_mr.lf_frame_hamiltonian(
        tmat, 0.0, 0.5, lam_orbit, shift_orbit
    )
    smat = lf_mr.lf_frame_overlap(lam_orbit, shift_orbit)
    energy, _, rank = lf_mr.lf_noci_lowest(hmat, smat)
    assert rank == 4
    assert energy == pytest.approx(-2.0, abs=1e-12)

@pytest.mark.parametrize("lam, shift", [
    (np.zeros((0, 0)), np.zeros(0)),
    (np.zeros((2, 3)), np.zeros(2)),
    (np.zeros((2, 2)), np.zeros(3)),
    (np.full((2, 2), np.nan), np.zeros(2)),
    (np.zeros((2, 2)), np.full(2, np.inf)),
])
def test_rejects_invalid_shapes_and_nonfinite_values(orbit, lam, shift):
    with pytest.raises(ValueError):
        orbit(lam, shift)


@pytest.mark.parametrize("target", ["lam", "shift"])
def test_rejects_unsupported_array_dtype(orbit, target):
    lam = np.zeros((2, 2))
    shift = np.zeros(2)
    if target == "lam":
        lam = lam.astype(np.float32)
    else:
        shift = shift.astype(np.float32)
    with pytest.raises(TypeError):
        orbit(lam, shift)
