"""Acceptance checks for the multi-electron LF frame Gram kernel."""

import numpy as np
import pytest

from src import lf_mr


@pytest.fixture
def overlap():
    if not hasattr(lf_mr, "lf_multi_frame_overlap"):
        pytest.skip("Student task pending: implement lf_multi_frame_overlap")
    function = lf_mr.lf_multi_frame_overlap
    try:
        function(np.zeros((1, 1, 1)), np.zeros((1, 1)), (1, 0))
    except NotImplementedError:
        pytest.skip("Student task pending: implement lf_multi_frame_overlap")
    return function


def test_one_alpha_electron_reduces_to_existing_frame_overlap(overlap):
    lam = np.array([
        [[0.2, -0.1, 0.3], [0.4, 0.1, -0.2], [-0.2, 0.3, 0.1]],
        [[-0.3, 0.2, 0.1], [0.1, -0.2, 0.4], [0.2, 0.1, -0.1]],
    ], dtype=np.float64)
    shift = np.array([[0.1, -0.2, 0.0], [-0.1, 0.1, 0.2]])
    result = overlap(lam, shift, (1, 0))
    assert result.shape == (2, 3, 1, 2, 3, 1)
    assert result.dtype == np.float64
    np.testing.assert_allclose(
        result[:, :, 0, :, :, 0], lf_mr.lf_frame_overlap(lam, shift),
        atol=2e-14, rtol=0,
    )
    beta_result = overlap(lam, shift, (0, 1))
    assert beta_result.shape == (2, 1, 3, 2, 1, 3)
    np.testing.assert_allclose(
        beta_result[:, 0, :, :, 0, :], lf_mr.lf_frame_overlap(lam, shift),
        atol=2e-14, rtol=0,
    )


def test_two_electron_coherent_displacements_and_electron_orthogonality(overlap):
    lam = np.zeros((2, 2, 2), dtype=np.float64)
    lam[1] = [[0.4, -0.2], [0.1, 0.3]]
    shift = np.zeros((2, 2), dtype=np.float64)
    result = overlap(lam, shift, (1, 1))
    assert result.shape == (2, 2, 2, 2, 2, 2)
    np.testing.assert_array_equal(
        result[0, :, :, 0, :, :].reshape(4, 4), np.eye(4),
    )
    np.testing.assert_array_equal(
        result[1, :, :, 1, :, :].reshape(4, 4), np.eye(4),
    )
    expected = np.array([
        [np.exp(-0.5 * (0.8**2 + 0.2**2)),
         np.exp(-0.5 * (0.2**2 + 0.4**2))],
        [np.exp(-0.5 * (0.2**2 + 0.4**2)),
         np.exp(-0.5 * (0.4**2 + 0.6**2))],
    ])
    for ia in range(2):
        for ib in range(2):
            assert result[0, ia, ib, 1, ia, ib] == pytest.approx(
                expected[ia, ib], abs=2e-14,
            )
    cross = result[0, :, :, 1, :, :].reshape(4, 4)
    np.testing.assert_array_equal(cross - np.diag(np.diag(cross)),
                                  np.zeros((4, 4)))
    matrix = result.reshape(8, 8)
    np.testing.assert_allclose(matrix, matrix.T, atol=2e-14, rtol=0)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-12


def test_other_filling_and_input_ownership(overlap):
    rng = np.random.default_rng(421)
    lam = rng.normal(scale=0.2, size=(2, 3, 3)).astype(np.float64)
    shift = rng.normal(scale=0.1, size=(2, 3)).astype(np.float64)
    lam_before, shift_before = lam.copy(), shift.copy()
    result = overlap(lam, shift, (2, 1))
    assert result.shape == (2, 3, 3, 2, 3, 3)
    for frame in range(2):
        np.testing.assert_allclose(
            result[frame, :, :, frame, :, :].reshape(9, 9),
            np.eye(9), atol=2e-14, rtol=0,
        )
    np.testing.assert_array_equal(lam, lam_before)
    np.testing.assert_array_equal(shift, shift_before)


@pytest.mark.parametrize("nelec", [(-1, 0), (3, 0), (1,), (True, 0)])
def test_rejects_invalid_electron_counts(overlap, nelec):
    with pytest.raises(ValueError):
        overlap(np.zeros((1, 2, 2)), np.zeros((1, 2)), nelec)


def test_rejects_incompatible_arrays(overlap):
    with pytest.raises(ValueError):
        overlap(np.zeros((1, 2, 2)), np.zeros((1, 3)), (1, 1))
    with pytest.raises(ValueError):
        overlap(np.full((1, 2, 2), np.nan), np.zeros((1, 2)), (1, 1))
    with pytest.raises(TypeError):
        overlap(np.zeros((1, 2, 2), dtype=np.float32),
                np.zeros((1, 2)), (1, 1))
