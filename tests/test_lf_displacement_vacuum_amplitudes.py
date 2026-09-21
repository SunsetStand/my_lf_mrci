"""Acceptance tests for LF vacuum-to-number-state displacement amplitudes."""

import numpy as np
import pytest

from src.lf_mp import (
    lf_displacement_vacuum_amplitudes,
    lf_franck_condon,
)


def test_vacuum_configuration_recovers_zero_phonon_franck_condon_matrix():
    lam = np.array(
        [
            [0.2, -0.3, 0.1],
            [0.0, 0.4, -0.2],
        ]
    )
    occupations = np.zeros((1, 2), dtype=int)

    amplitudes = lf_displacement_vacuum_amplitudes(lam, occupations)

    assert amplitudes.shape == (1, 3, 3)
    assert np.issubdtype(amplitudes.dtype, np.floating)
    np.testing.assert_allclose(amplitudes[0], lf_franck_condon(lam), atol=1e-14)


def test_one_mode_values_preserve_odd_power_sign():
    lam = np.array([[0.2, -0.3]])
    occupations = np.array([[0], [1], [2]])
    delta = 0.5
    gaussian = np.exp(-0.5 * delta**2)

    amplitudes = lf_displacement_vacuum_amplitudes(lam, occupations)

    np.testing.assert_allclose(
        amplitudes[:, 0, 1],
        gaussian * np.array([1.0, delta, delta**2 / np.sqrt(2.0)]),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        amplitudes[:, 1, 0],
        gaussian * np.array([1.0, -delta, delta**2 / np.sqrt(2.0)]),
        atol=1e-14,
    )


def test_multimode_amplitude_is_product_of_signed_mode_factors():
    lam = np.diag([0.3, 0.4])
    occupations = np.array([[1, 1], [2, 1]])
    delta = np.array([0.3, -0.4])
    gaussian = np.exp(-0.5 * np.dot(delta, delta))

    amplitudes = lf_displacement_vacuum_amplitudes(lam, occupations)

    np.testing.assert_allclose(
        amplitudes[:, 0, 1],
        gaussian
        * np.array(
            [
                delta[0] * delta[1],
                delta[0] ** 2 * delta[1] / np.sqrt(2.0),
            ]
        ),
        atol=1e-14,
    )


def test_diagonal_displacement_cannot_create_phonons():
    lam = np.diag([0.7, -0.4])
    occupations = np.array([[0, 0], [1, 0], [0, 2], [1, 1]])

    amplitudes = lf_displacement_vacuum_amplitudes(lam, occupations)

    np.testing.assert_allclose(np.diagonal(amplitudes[0]), 1.0, atol=1e-14)
    np.testing.assert_allclose(
        np.diagonal(amplitudes[1:], axis1=1, axis2=2),
        0.0,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    ("lam", "occupations"),
    [
        (np.zeros(2), np.zeros((1, 2), dtype=int)),
        (np.zeros((2, 2)), np.zeros(2, dtype=int)),
        (np.zeros((2, 2)), np.zeros((1, 3), dtype=int)),
        (np.zeros((2, 2)), np.array([[0.0, 1.0]])),
        (np.zeros((2, 2)), np.array([[0, -1]])),
        (np.array([[0.0, np.nan]]), np.zeros((1, 1), dtype=int)),
    ],
)
def test_rejects_invalid_shapes_values_and_occupation_dtype(lam, occupations):
    with pytest.raises(ValueError):
        lf_displacement_vacuum_amplitudes(lam, occupations)
