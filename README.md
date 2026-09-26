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

Run the extended Fig. 2b scan containing LF-MP2 with:

```bash
python -m scripts.fig2b_mp2_scan
python -m scripts.plot_fig2b_mp2
```

This writes `data/fig2b_mp2.csv`, `figures/fig2b_mp2.png`, and
`figures/fig2b_mp2.pdf` without replacing the HF-only artifacts above.  The
CSV preserves all previous CS-HF, CS-MP2, and LF-HF fields and adds the
pure-phonon, electronic-single, and total LF-MP2 corrections.  LF-MP2 uses a
collective total-phonon cutoff `max_total=10`, corresponding to 1000
non-vacuum configurations for four modes.  The paper reports a cutoff of 16;
against that value, the maximum total-energy difference at representative
points `alpha=0.4, 2.2, 2.4, 3.0` was `6.3e-15`.  The automated convergence
test checks the worst anchor at `alpha=2.2` to an absolute tolerance of
`1e-12`.

The LF-MP2 energies were also checked against the vector markers in the
published Fig. 2b.  Digitizing all 16 plotted coupling values gives a maximum
absolute difference of `6.4e-4` and an RMS difference of `2.2e-4`, both below
the energy resolution set by the published line width.  This is a
figure-resolution comparison because the article does not provide the
underlying numerical table.

Current learning material: `docs/cs_lf_theory_stage.md`.  The Fig. 2d script
`scripts/exact_fig2d_probe.py` is deliberately a finite-cutoff diagnostic, not
a converged reproduction of the published strong-coupling curve.

Run the fixed-frame MR-LF NOCI baseline and add it to the Fig. 2b comparison:

```bash
python -m scripts.fig2b_mr_scan
python -m scripts.plot_fig2b_mr
```

This writes `data/fig2b_mr.csv` and `figures/fig2b_mr.{png,pdf}` without
replacing the earlier scan or figure. At each of the same 16 alpha values,
the recipe combines the full translation orbit of the lowest LF-HF frame
with the full orbit of the frame optimized from the coherent-state start.
The figure retains the exact ED, CS-HF/MP2, and LF-HF/MP2 curves and adds
the MR-LF NOCI energy and electronic density imbalance. This is a fixed
variational frame recipe, not automatic reference selection or CI with
external phonon excitations.

To examine how the number of LF frames and the retained overlap rank affect
the energy at the coupling where LF-HF differs most from exact ED, run:

```bash
python -m scripts.fig2b_mr_convergence
python -m scripts.plot_fig2b_mr_convergence
```

The scripts select `alpha=2.4` from the existing HF and exact CSV files.
They start from one LF-HF frame, restore its four-site translation orbit,
add the coherent-state-start orbit, and then greedily add complete orbits
from a fixed displacement path with theta values
`0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2.0`. The output is
`data/fig2b_mr_worst_convergence.csv` and
`figures/fig2b_mr_worst_convergence.{png,pdf}`. The plot shows exact ED,
nominal versus retained basis dimension, and an overlap-cut sensitivity
range. This finite candidate pool leaves a nonzero gap to exact; its
late points depend on the overlap threshold and should not be presented
as convergence to the exact state.
