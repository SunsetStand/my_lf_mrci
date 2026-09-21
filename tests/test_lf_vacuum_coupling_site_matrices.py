"""Acceptance tests for non-vacuum LF couplings in the site basis."""

import numpy as np
import pytest

from src.lf_mp import lf_vacuum_coupling_site_matrices


def test_one_site_separates_single_phonon_residual_from_higher_excitations():
    tmat = np.array([[2.0]])
    g = 0.6
    omega = 0.5
    shift = np.array([-0.2])
    lam = np.array([[0.7]])
    occupations = np.array([[1], [2], [3]])

    couplings = lf_vacuum_coupling_site_matrices(
        tmat, g, omega, shift, lam, occupations
    )

    residual = omega * shift[0] + g - omega * lam[0, 0]
    assert couplings.shape == (3, 1, 1)
    np.testing.assert_allclose(couplings[:, 0, 0], [residual, 0.0, 0.0], atol=1e-14)


def test_two_site_local_lf_combines_signed_hopping_and_linear_residual():
    tmat = np.array([[0.0, -1.0], [-1.0, 0.0]])
    g = 0.5
    omega = 0.7
    shift = np.array([0.1, -0.2])
    lam = np.diag([0.3, 0.4])
    occupations = np.array([[1, 0], [0, 1], [1, 1], [2, 0]])
    gaussian = np.exp(-0.5 * (0.3**2 + 0.4**2))

    couplings = lf_vacuum_coupling_site_matrices(
        tmat, g, omega, shift, lam, occupations
    )

    expected = np.array(
        [
            [
                [omega * shift[0] + g - omega * lam[0, 0], -0.3 * gaussian],
                [0.3 * gaussian, omega * shift[0]],
            ],
            [
                [omega * shift[1], 0.4 * gaussian],
                [-0.4 * gaussian, omega * shift[1] + g - omega * lam[1, 1]],
            ],
            [[0.0, 0.12 * gaussian], [0.12 * gaussian, 0.0]],
            [
                [0.0, -(0.3**2) * gaussian / np.sqrt(2.0)],
                [-(0.3**2) * gaussian / np.sqrt(2.0), 0.0],
            ],
        ]
    )
    np.testing.assert_allclose(couplings, expected, atol=1e-14)
    assert not np.allclose(couplings[0], couplings[0].T)


def test_zero_transformation_and_zero_coupling_give_zero_nonvacuum_couplings():
    tmat = np.array([[0.0, -1.0], [-1.0, 0.0]])
    occupations = np.array([[1, 0], [0, 2], [1, 1]])

    couplings = lf_vacuum_coupling_site_matrices(
        tmat,
        g=0.0,
        omega=0.5,
        shift=np.zeros(2),
        lam=np.zeros((2, 2)),
        occupations=occupations,
    )

    np.testing.assert_allclose(couplings, 0.0, atol=1e-14)


def test_complex_hopping_dtype_is_preserved():
    tmat = np.array([[0.0, 1.0j], [-1.0j, 0.0]])

    couplings = lf_vacuum_coupling_site_matrices(
        tmat,
        g=0.0,
        omega=0.5,
        shift=np.zeros(2),
        lam=np.diag([0.2, 0.3]),
        occupations=np.array([[1, 0]]),
    )

    assert np.issubdtype(couplings.dtype, np.complexfloating)
    assert couplings[0, 0, 1].imag != 0.0


@pytest.mark.parametrize(
    ("tmat", "shift", "lam", "occupations"),
    [
        (np.zeros((2, 3)), np.zeros(2), np.zeros((2, 2)), np.array([[1, 0]])),
        (np.zeros((2, 2)), np.zeros(2), np.zeros(2), np.array([[1, 0]])),
        (np.zeros((2, 2)), np.zeros(3), np.zeros((2, 2)), np.array([[1, 0]])),
        (np.zeros((2, 2)), np.zeros(2), np.zeros((1, 2)), np.array([[1]])),
        (np.zeros((2, 2)), np.zeros(2), np.zeros((2, 2)), np.array([[0, 0]])),
        (
            np.array([[0.0, 1.0], [0.0, 0.0]]),
            np.zeros(2),
            np.zeros((2, 2)),
            np.array([[1, 0]]),
        ),
        (
            np.zeros((2, 2)),
            np.array([0.0, np.nan]),
            np.zeros((2, 2)),
            np.array([[1, 0]]),
        ),
    ],
)
def test_rejects_incompatible_dimensions_and_vacuum_configurations(
    tmat, shift, lam, occupations
):
    with pytest.raises(ValueError):
        lf_vacuum_coupling_site_matrices(
            tmat, g=0.4, omega=0.5, shift=shift, lam=lam, occupations=occupations
        )


@pytest.mark.parametrize(("g", "omega"), [(np.nan, 0.5), (0.4, np.nan), (0.4, 0.0)])
def test_rejects_invalid_scalar_parameters(g, omega):
    with pytest.raises(ValueError):
        lf_vacuum_coupling_site_matrices(
            np.zeros((1, 1)),
            g=g,
            omega=omega,
            shift=np.zeros(1),
            lam=np.zeros((1, 1)),
            occupations=np.ones((1, 1), dtype=int),
        )
