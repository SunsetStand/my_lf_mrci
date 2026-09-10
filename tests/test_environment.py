"""Environment checks only; no model implementation."""

import importlib

import numpy as np
from pyscf.fci import cistring


def test_required_imports():
    for name in (
        "numpy", "scipy", "pyscf", "pyscf.lib", "pyscf.ao2mo",
        "pyscf.fci.cistring", "pyscf.fci.rdm", "pyscf.fci.direct_ep",
        "matplotlib", "pytest",
    ):
        importlib.import_module(name)
    # Compatibility probe only: never use this private helper in model code.
    module = importlib.import_module("pyscf.fci.direct_spin1")
    assert callable(module._unpack_nelec)


def test_cistring_basis_and_links():
    assert cistring.num_strings(4, 1) == 4
    strings = cistring.make_strings(range(4), 1)
    np.testing.assert_array_equal(strings, [1, 2, 4, 8])
    links = cistring.gen_linkstr_index(range(4), 1)
    assert links.shape == (4, 4, 4)
    for source, rows in enumerate(links):
        for a, i, target, sign in rows:
            assert strings[source] == 1 << i
            assert strings[target] == 1 << a
            assert sign == 1


def test_matplotlib_headless(tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    output = tmp_path / "smoke.png"
    fig.savefig(output)
    plt.close(fig)
    assert output.stat().st_size > 0
