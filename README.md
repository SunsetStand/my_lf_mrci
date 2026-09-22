# MR_LF learning environment

Use WSL2 (`ubuntu-d`) and open `/mnt/d/code/MR_LF` in VS Code.
The existing `.venv` belongs to Windows Python 3.9.10; leave it untouched.
Linux interpreter: `/home/enovo/.venvs/mr_lf/bin/python` (Python 3.12.3).
`requirements.txt` records the installed direct and transitive package versions.

```bash
cd /mnt/d/code/MR_LF
source /home/enovo/.venvs/mr_lf/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m pytest -q
ruff check --no-cache tests/test_environment.py
ruff check --no-cache src tests
```

In VS Code, run `Python: Select Interpreter` and select the Linux interpreter
above if a previous workspace selection overrides the default setting.
The student's initial `src/my_direct_ep.py` contains an unused NumPy import;
the full Ruff check may report F401 until that import is used.

Tests cover the environment, basis conventions, Hubbard–Holstein contractions,
tiny independent dense references, Davidson solves, scans, and headless plots.
The private `_unpack_nelec` helper is probed only for compatibility.

Run the reproducible CS-HF/CS-MP2/LF-HF part of the Fig. 2b scan and plot with:

```bash
python -m scripts.fig2b_hf_scan
python -m scripts.plot_fig2b_hf
```

The scan uses ``alpha = 0.0, 0.2, ..., 3.0`` and the paper's full
density-diagonal LF parameters ``lam[x, p]``.  Each LF-HF point includes a
coherent-state start, one localized start per site, and reproducible uniform
random starts.  It writes
`data/fig2b_hf.csv`, `figures/fig2b_hf.png`, and
`figures/fig2b_hf.pdf`.  The CSV includes all four site densities and the
optimization diagnostics needed to audit the LF-HF symmetry breaking, along
with the CS-MP2 correction and total energy.  The energy panel also reads
`data/fig2b_exact.csv` and overlays the existing exact ED reference.

Current learning material: `docs/cs_lf_theory_stage.md`.  The Fig. 2d script
`scripts/exact_fig2d_probe.py` is deliberately a finite-cutoff diagnostic, not
a converged reproduction of the published strong-coupling curve.
