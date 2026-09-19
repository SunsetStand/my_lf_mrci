"""Acceptance tests for a single local LF-HF coupling point."""

import numpy as np
import pytest

from src.lf_mp import lf_hf_local_alpha_point
from src.my_direct_ep import electron_ring_hopping


@pytest.mark.parametrize(
    ("alpha", "expected_energy", "expected_imbalance"),
    [
        (0.0, -2.0, 0.0),
        (2.2, -2.82487018210281, 0.0),
        (2.4, -2.924947166702873, 0.7391966505886146),
    ],
)
def test_reports_energy_density_and_symmetry_diagnostics(
    alpha,
    expected_energy,
    expected_imbalance,
):
    hopping = electron_ring_hopping(4, -1.0)
    omega = 0.5

    best, results = lf_hf_local_alpha_point(alpha, hopping, omega)

    assert any(best is result for result in results)
    assert best.success
    assert best.alpha == alpha
    np.testing.assert_allclose(best.g, np.sqrt(alpha * omega), atol=0.0)
    np.testing.assert_allclose(best.fun, expected_energy, atol=1e-11)
    np.testing.assert_allclose(best.density, np.abs(best.coeff) ** 2, atol=1e-14)
    np.testing.assert_allclose(np.sum(best.density), 1.0, atol=1e-14)
    np.testing.assert_allclose(best.density_imbalance, expected_imbalance, atol=1e-8)
    assert best.nstart == 9
    assert best.nconverged == sum(bool(result.success) for result in results)
    assert 1 <= best.nconverged <= best.nstart


def test_forwards_start_count_to_existing_drivers():
    hopping = electron_ring_hopping(4, -1.0)

    best, results = lf_hf_local_alpha_point(
        0.0,
        hopping,
        0.5,
        nrandom=0,
    )

    assert len(results) == 1
    assert best.nstart == 1
    assert best.nconverged == 1


@pytest.mark.parametrize("alpha", [-0.1, np.nan, np.inf, 1.0j])
def test_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        lf_hf_local_alpha_point(alpha, np.zeros((4, 4)), omega=0.5)


@pytest.mark.parametrize("omega", [0.0, -0.5, np.nan, np.inf, 1.0j])
def test_rejects_invalid_frequency(omega):
    with pytest.raises(ValueError):
        lf_hf_local_alpha_point(0.8, np.zeros((4, 4)), omega)


@pytest.mark.parametrize("tmat", [np.zeros(4), np.zeros((3, 4))])
def test_rejects_invalid_hopping_shape(tmat):
    with pytest.raises(ValueError):
        lf_hf_local_alpha_point(0.8, tmat, omega=0.5)
