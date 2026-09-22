"""Acceptance tests for the one-electron LF-MP2 energy correction."""

import numpy as np
import pytest

from src.lf_mp import lf_mp2_energy


def test_combines_complex_pure_and_single_channels_without_prefactors():
    pure_matrix_elements = np.array([1.0 + 2.0j, 2.0 - 1.0j])
    pure_denominators = np.array([-2.0, -4.0])
    single_matrix_elements = np.array(
        [
            [[1.0j, 2.0]],
            [[0.5 + 0.5j, 0.0]],
        ]
    )
    single_denominators = np.array(
        [
            [[-1.0, -2.0]],
            [[-4.0, -5.0]],
        ]
    )
    expected = np.sum(np.abs(pure_matrix_elements) ** 2 / pure_denominators)
    expected += np.sum(
        np.abs(single_matrix_elements) ** 2 / single_denominators
    )

    correction = lf_mp2_energy(
        pure_matrix_elements,
        single_matrix_elements,
        pure_denominators,
        single_denominators,
    )

    assert isinstance(correction, float)
    assert correction == pytest.approx(expected, abs=1e-14)
    assert correction < 0.0


def test_zero_matrix_elements_give_exactly_zero_correction():
    correction = lf_mp2_energy(
        np.zeros(2, dtype=complex),
        np.zeros((2, 1, 3), dtype=complex),
        np.array([-0.5, -1.0]),
        -np.ones((2, 1, 3)),
    )

    assert correction == 0.0


@pytest.mark.parametrize(
    (
        "pure_matrix_elements",
        "single_matrix_elements",
        "pure_denominators",
        "single_denominators",
    ),
    [
        (np.zeros((2, 1)), np.zeros((2, 1, 1)), -np.ones(2), -np.ones((2, 1, 1))),
        (np.zeros(2), np.zeros((2, 1)), -np.ones(2), -np.ones((2, 1, 1))),
        (np.zeros(2), np.zeros((2, 1, 1)), -np.ones((2, 1)), -np.ones((2, 1, 1))),
        (np.zeros(2), np.zeros((2, 1, 1)), -np.ones(2), -np.ones((2, 1))),
        (np.zeros(1), np.zeros((2, 1, 1)), -np.ones(1), -np.ones((2, 1, 1))),
        (np.zeros(2), np.zeros((2, 1, 1)), -np.ones(1), -np.ones((2, 1, 1))),
        (np.zeros(2), np.zeros((2, 1, 2)), -np.ones(2), -np.ones((2, 1, 1))),
    ],
)
def test_rejects_invalid_dimensions_and_incompatible_shapes(
    pure_matrix_elements,
    single_matrix_elements,
    pure_denominators,
    single_denominators,
):
    with pytest.raises(ValueError):
        lf_mp2_energy(
            pure_matrix_elements,
            single_matrix_elements,
            pure_denominators,
            single_denominators,
        )


@pytest.mark.parametrize(
    (
        "pure_matrix_elements",
        "single_matrix_elements",
        "pure_denominators",
        "single_denominators",
    ),
    [
        (np.array([np.nan]), np.zeros((1, 1, 1)), np.array([-1.0]), -np.ones((1, 1, 1))),
        (np.zeros(1), np.array([[[np.inf]]]), np.array([-1.0]), -np.ones((1, 1, 1))),
        (np.zeros(1), np.zeros((1, 1, 1)), np.array([0.0]), -np.ones((1, 1, 1))),
        (np.zeros(1), np.zeros((1, 1, 1)), np.array([-1.0]), np.ones((1, 1, 1))),
        (
            np.zeros(1),
            np.zeros((1, 1, 1)),
            np.array([-1.0 + 0.1j]),
            -np.ones((1, 1, 1)),
        ),
        (np.zeros(1), np.zeros((1, 1, 1)), np.array([-1.0]), np.array([[[-np.inf]]])),
    ],
)
def test_rejects_nonfinite_elements_and_invalid_denominators(
    pure_matrix_elements,
    single_matrix_elements,
    pure_denominators,
    single_denominators,
):
    with pytest.raises(ValueError):
        lf_mp2_energy(
            pure_matrix_elements,
            single_matrix_elements,
            pure_denominators,
            single_denominators,
        )
