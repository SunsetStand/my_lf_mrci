"""Acceptance tests for the one-electron LF effective Hamiltonian."""

import numpy as np
import pytest

from src.cs_mp import cs_fock
from src.lf_mp import lf_effective_one_body
from src.my_direct_ep import electron_ring_hopping


def test_zero_lambda_recovers_cs_fock_without_global_boson_constant():
    hopping = electron_ring_hopping(4, -1.0)
    shift = np.array([-0.2, -0.1, 0.0, 0.1])
    lam = np.zeros((4, 4))

    effective = lf_effective_one_body(
        hopping,
        g=0.4,
        omega=0.5,
        shift=shift,
        lam=lam,
    )

    np.testing.assert_allclose(
        effective,
        cs_fock(hopping, 0.4, shift),
        atol=1e-14,
    )


def test_canonical_local_lf_shift_dresses_hopping_and_polaron_energy():
    hopping = electron_ring_hopping(4, -1.0)
    omega = 0.5
    g = 0.4
    displacement = g / omega
    lam = displacement * np.eye(4)
    overlap = np.full((4, 4), np.exp(-(displacement**2)))
    np.fill_diagonal(overlap, 1.0)
    expected = hopping * overlap - (g**2 / omega) * np.eye(4)

    effective = lf_effective_one_body(
        hopping,
        g,
        omega,
        np.zeros(4),
        lam,
    )

    np.testing.assert_allclose(effective, expected, atol=1e-14)


def test_complex_hermitian_hopping_remains_hermitian_and_unmodified():
    phase = 0.31
    hopping = np.array(
        [
            [0.2, -np.exp(1j * phase)],
            [-np.exp(-1j * phase), -0.1],
        ],
        dtype=np.complex128,
    )
    shift = np.array([0.2, -0.3])
    lam = np.array([[0.1, 0.4], [-0.2, 0.3]])
    original_hopping = hopping.copy()
    original_shift = shift.copy()
    original_lam = lam.copy()

    effective = lf_effective_one_body(hopping, 0.4, 0.5, shift, lam)

    assert effective.dtype == np.complex128
    np.testing.assert_allclose(effective, effective.conj().T, atol=1e-14)
    np.testing.assert_array_equal(hopping, original_hopping)
    np.testing.assert_array_equal(shift, original_shift)
    np.testing.assert_array_equal(lam, original_lam)


def test_gauge_translation_changes_effective_matrix_by_scalar_identity():
    rng = np.random.default_rng(12)
    hopping = electron_ring_hopping(4, -1.0)
    shift = rng.normal(size=4)
    lam = rng.normal(size=(4, 4))
    translation = rng.normal(size=4)
    omega = 0.5

    effective = lf_effective_one_body(hopping, 0.4, omega, shift, lam)
    translated = lf_effective_one_body(
        hopping,
        0.4,
        omega,
        shift + translation,
        lam + translation[:, None],
    )
    scalar_change = -omega * (
        2.0 * np.dot(shift, translation)
        + np.dot(translation, translation)
    )

    np.testing.assert_allclose(
        translated,
        effective + scalar_change * np.eye(4),
        atol=1e-13,
    )


@pytest.mark.parametrize(
    ("tmat", "shift", "lam"),
    [
        (np.zeros(3), np.zeros(3), np.zeros((3, 3))),
        (np.zeros((2, 3)), np.zeros(2), np.zeros((2, 3))),
        (np.zeros((3, 3)), np.zeros(3), np.zeros((2, 3))),
        (np.zeros((3, 3)), np.zeros((1, 3)), np.zeros((3, 3))),
        (np.zeros((3, 3)), np.zeros(2), np.zeros((3, 3))),
    ],
)
def test_rejects_incompatible_shapes(tmat, shift, lam):
    with pytest.raises(ValueError):
        lf_effective_one_body(tmat, 0.4, 0.5, shift, lam)


@pytest.mark.parametrize("omega", [0.0, -0.5])
def test_rejects_nonpositive_frequency(omega):
    with pytest.raises(ValueError, match="positive"):
        lf_effective_one_body(
            np.zeros((3, 3)),
            0.4,
            omega,
            np.zeros(3),
            np.zeros((3, 3)),
        )
