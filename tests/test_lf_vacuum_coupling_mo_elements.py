"""Acceptance tests for the LF site-to-canonical-MO coupling transform."""

import numpy as np
import pytest

from src.lf_mp import lf_vacuum_coupling_mo_elements


def test_identity_mos_expose_occupied_trace_and_ai_axis_order():
    site_couplings = np.empty((2, 4, 4), dtype=complex)
    for config in range(2):
        for p in range(4):
            for q in range(4):
                site_couplings[config, p, q] = 100 * config + 10 * p + q + 0.25j

    mo_couplings, pure_phonon, singles = lf_vacuum_coupling_mo_elements(
        site_couplings, np.eye(4), nocc=2
    )

    assert mo_couplings.shape == (2, 4, 4)
    assert pure_phonon.shape == (2,)
    assert singles.shape == (2, 2, 2)
    np.testing.assert_allclose(mo_couplings, site_couplings, atol=1e-14)
    for config in range(2):
        assert pure_phonon[config] == pytest.approx(
            site_couplings[config, 0, 0] + site_couplings[config, 1, 1]
        )
        for occupied in range(2):
            for virtual_offset in range(2):
                virtual_mo = 2 + virtual_offset
                assert singles[config, occupied, virtual_offset] == pytest.approx(
                    site_couplings[config, virtual_mo, occupied]
                )


def test_complex_mo_transform_uses_conjugate_transpose():
    site_couplings = np.array(
        [
            [[1.0, 2.0 - 1.0j], [-0.5 + 0.2j, 3.0]],
            [[0.3j, -1.2], [0.7 + 0.4j, 2.1]],
        ]
    )
    mo_coeff = np.array([[1.0, 1.0], [1.0j, -1.0j]]) / np.sqrt(2.0)

    mo_couplings, pure_phonon, singles = lf_vacuum_coupling_mo_elements(
        site_couplings, mo_coeff, nocc=1
    )

    expected = np.stack(
        [mo_coeff.conj().T @ matrix @ mo_coeff for matrix in site_couplings]
    )
    np.testing.assert_allclose(mo_couplings, expected, atol=1e-14)
    np.testing.assert_allclose(pure_phonon, expected[:, 0, 0], atol=1e-14)
    np.testing.assert_allclose(singles[:, 0, 0], expected[:, 1, 0], atol=1e-14)


def test_one_electron_shapes_keep_configuration_and_virtual_axes_distinct():
    site_couplings = np.arange(48.0).reshape(3, 4, 4)

    _, pure_phonon, singles = lf_vacuum_coupling_mo_elements(
        site_couplings, np.eye(4), nocc=1
    )

    assert pure_phonon.shape == (3,)
    assert singles.shape == (3, 1, 3)
    np.testing.assert_allclose(singles[:, 0, :], site_couplings[:, 1:, 0], atol=1e-14)


@pytest.mark.parametrize(
    ("site_couplings", "mo_coeff", "nocc"),
    [
        (np.zeros((2, 3)), np.eye(3), 1),
        (np.zeros((2, 3, 4)), np.eye(3), 1),
        (np.full((2, 3, 3), np.nan), np.eye(3), 1),
        (np.zeros((2, 3, 3)), np.zeros((3, 2)), 1),
        (np.zeros((2, 3, 3)), np.diag([1.0, 1.0, 2.0]), 1),
        (np.zeros((2, 3, 3)), np.eye(3), 0),
        (np.zeros((2, 3, 3)), np.eye(3), 3),
    ],
)
def test_rejects_invalid_shapes_nonunitary_mos_and_occupation_count(
    site_couplings, mo_coeff, nocc
):
    with pytest.raises(ValueError):
        lf_vacuum_coupling_mo_elements(site_couplings, mo_coeff, nocc)


@pytest.mark.parametrize("nocc", [1.5, True])
def test_rejects_noninteger_occupation_count_with_type_error(nocc):
    with pytest.raises(TypeError):
        lf_vacuum_coupling_mo_elements(np.zeros((2, 3, 3)), np.eye(3), nocc)
