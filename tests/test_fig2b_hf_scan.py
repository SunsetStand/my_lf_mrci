"""Acceptance tests for the in-memory Fig. 2b HF scan."""

import numpy as np
import pytest

from scripts import fig2b_hf_scan as scan

EXPECTED_KEYS = {
    "alpha",
    "g",
    "cs_energy",
    "cs_mp2_correction",
    "cs_mp2_energy",
    "cs_density",
    "cs_density_imbalance",
    "cs_niter",
    "cs_density_residual",
    "lf_energy",
    "lf_density",
    "lf_density_imbalance",
    "lf_nstart",
    "lf_nconverged",
    "lf_shift_residual",
}


def test_real_scan_recovers_cs_line_and_lf_symmetry_breaking():
    alpha_values = np.array([0.0, 2.2, 2.4])

    records = scan.run_hf_scan(alpha_values)

    assert len(records) == 3
    assert all(set(record) == EXPECTED_KEYS for record in records)
    np.testing.assert_array_equal([record["alpha"] for record in records], alpha_values)
    np.testing.assert_allclose(
        [record["cs_energy"] for record in records],
        -2.0 - alpha_values / 4.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        [record["cs_mp2_correction"] for record in records],
        -23.0 * alpha_values / 180.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        [record["cs_mp2_energy"] for record in records],
        -2.0 - 17.0 * alpha_values / 45.0,
        atol=1e-14,
    )
    for record in records:
        assert record["cs_mp2_energy"] == pytest.approx(
            record["cs_energy"] + record["cs_mp2_correction"],
            abs=1e-14,
        )
        assert record["lf_energy"] <= record["cs_energy"] + 1e-12
        assert record["cs_density"].shape == (4,)
        assert record["lf_density"].shape == (4,)
        np.testing.assert_allclose(record["cs_density"], 0.25, atol=1e-14)
        np.testing.assert_allclose(np.sum(record["lf_density"]), 1.0, atol=1e-14)
        assert record["lf_nstart"] == 9
        assert 1 <= record["lf_nconverged"] <= record["lf_nstart"]
    np.testing.assert_allclose(records[1]["lf_density_imbalance"], 0.0, atol=1e-8)
    np.testing.assert_allclose(records[2]["lf_density_imbalance"], 0.7391966505886146, atol=1e-8)


def test_preserves_unsorted_and_repeated_input_order():
    alpha_values = np.array([0.4, 0.0, 0.4])

    records = scan.run_hf_scan(alpha_values, nrandom=0)

    np.testing.assert_array_equal([record["alpha"] for record in records], alpha_values)


@pytest.mark.parametrize(
    "alpha_values",
    [
        np.array([]),
        np.zeros((1, 2)),
        np.array([-0.1]),
        np.array([np.nan]),
        np.array([np.inf]),
        np.array([1.0j]),
    ],
)
def test_rejects_invalid_alpha_arrays(alpha_values):
    with pytest.raises(ValueError):
        scan.run_hf_scan(alpha_values)


def test_reports_alpha_when_cs_hf_fails(monkeypatch):
    def fake_cs_hf_scf(*args, **kwargs):
        return 0.0, np.full(4, 0.5), np.zeros(4), False, 1, 1.0

    monkeypatch.setattr(scan.cs_mp, "cs_hf_scf", fake_cs_hf_scf)

    with pytest.raises(RuntimeError, match=r"alpha=0\.8"):
        scan.run_hf_scan(np.array([0.8]), nrandom=0)
