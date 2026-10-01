"""Compare fixed-frame four-electron LF-NOCI with audited Fig. 2d data.

Run from the repository root with python -m scripts.fig2d_mr_scan.
This driver calls the accepted model kernels; it does not optimize MR frames.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pyscf import lib

from src import lf_mr, lf_mp, my_direct_ep as ep

ROOT = Path(__file__).resolve().parents[1]
L, NELEC, U, T = 4, (2, 2), 4.0, -1.0
CUT = 1e-10
RECIPE = 'selected_HF_translation_orbit_plus_uniform_CS;full_site_CI_per_frame'


def _key(row):
    return float(row['omega']), round(float(row['alpha']), 10)


def _read(path):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    result = {_key(r): r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate points in {path}')
    return result


def read_sources():
    exact = _read(ROOT / 'data/fig2d_exact.csv')
    points = _read(ROOT / 'data/fig2d_lf.csv')
    expected = {(w, round(a, 10)) for w in (0.5, 5.0) for a in np.linspace(0, 4, 11)}
    if set(exact) != expected or set(points) != expected:
        raise ValueError('Source grids must match the complete 22-point Fig. 2d grid')
    for key, r in exact.items():
        if (int(r['L']) != L or (int(r['neleca']), int(r['nelecb'])) != NELEC
                or float(r['U']) != U or float(r['t']) != T
                or r['convention'] != 'centered_solve_paper_energy'
                or r['cutoff_converged'] != 'True' or r['solver_converged'] != 'True'
                or not np.isfinite(float(r['residual']))
                or float(r['residual']) >= float(r['residual_tolerance'])):
            raise ValueError(f'Incompatible/unconverged ED metadata: {key}')
        w, a = key
        if not np.isclose(float(r['g'])**2, a*w, atol=1e-12, rtol=0):
            raise ValueError('ED coupling mismatch')
        shift = -a*sum(NELEC)**2/L
        if (not np.isclose(float(r['paper_energy_shift']), shift, atol=1e-10, rtol=0)
                or not np.isclose(float(r['energy']), float(r['energy_centered'])+shift,
                                  atol=1e-10, rtol=0)
                or abs(float(points[key]['ed_energy'])-float(r['energy'])) > 1e-10):
            raise ValueError('ED energy convention mismatch')
        if a > 0 and (not r['delta_E']
                       or float(r['delta_E']) >= float(r['cutoff_tolerance'])):
            raise ValueError('ED final cutoff step exceeds tolerance')
    with np.load(ROOT / 'data/fig2d_lf_references.npz') as data:
        refs = {key: data[key].copy() for key in data.files}
    ref_index = {(float(w), round(float(a), 10)): i
                 for i, (w, a) in enumerate(zip(refs['omega'], refs['alpha']))}
    if len(ref_index) != 22 or set(ref_index) != expected:
        raise ValueError('Saved reference grid mismatch')
    return exact, points, refs, ref_index


def measure(tmat, g, omega, lam, shift, occ):
    """Solve accepted kernels and compute diagnostic expectations in their S metric."""
    H = lf_mr.lf_multi_frame_hamiltonian(tmat, U, g, omega, lam, shift, NELEC)
    S = lf_mr.lf_multi_frame_overlap(lam, shift, NELEC)
    energy, coeff, rank = lf_mr.lf_noci_lowest(H, S, overlap_cut=CUT)
    c = coeff.ravel()
    h, s = H.reshape(c.size, c.size), S.reshape(c.size, c.size)
    sc = s @ c
    residual = h @ c - energy*sc
    eigenvalues, vectors = np.linalg.eigh(s)
    keep = eigenvalues > CUT*max(1.0, eigenvalues[-1])
    X = vectors[:, keep]/np.sqrt(eigenvalues[keep])
    spectrum = np.linalg.eigvalsh(X.T @ h @ X)
    # Sum over frames in each orthogonal electronic determinant block.
    weights = np.sum(coeff*sc.reshape(coeff.shape), axis=0)
    density = np.stack((weights.sum(axis=1) @ occ, weights.sum(axis=0) @ occ))
    checks = [lf_mr.lf_noci_lowest(H, S, overlap_cut=cut)
              for cut in (1e-8, 1e-12)]
    result = dict(
        energy=energy, rank=rank, nframes=len(lam), matrix_dimension=c.size,
        norm_error=abs(c@sc-1), residual_full=float(np.linalg.norm(residual)),
        residual_projected=float(np.linalg.norm(X.T@residual)),
        gap=float(spectrum[1]-spectrum[0]),
        cutoff_energy_spread=float(np.ptp([energy]+[v[0] for v in checks])),
        rank_loose=checks[0][2], rank_tight=checks[1][2],
    )
    if (result['norm_error'] > 1e-9 or result['residual_projected'] > 1e-8
            or result['cutoff_energy_spread'] > 1e-7
            or not np.allclose(density.sum(axis=1), NELEC, atol=1e-9, rtol=0)
            or weights.min() < -1e-9):
        raise RuntimeError(f'Invalid MR diagnostics: {result}')
    return result, coeff, density


def run_scan():
    exact, points, refs, indices = read_sources()
    tmat = ep.electron_ring_hopping(L, T)
    strings, _ = ep.make_electron_basis(L, NELEC[0])
    occ = ((strings[:, None] >> np.arange(L)) & 1).astype(np.float64)
    records, states = [], {}
    for key in sorted(points):
        omega, alpha = key
        p, ed, i = points[key], exact[key], indices[key]
        g = np.sqrt(alpha*omega)
        coeff, lam0 = refs['mo_coeff'][i], refs['lam'][i]
        hf, rho_hf = lf_mp.lf_hf_multi_fixed_energy(
            tmat, U, g, omega, NELEC, coeff[0, :, :2], coeff[1, :, :2],
            lam0, np.zeros(L),
        )
        if (abs(hf-float(p['hf_energy'])) > 1e-8
                or not np.allclose(rho_hf, refs['spin_density'][i], atol=1e-9, rtol=0)
                or max(float(p['orbital_grad_norm']), float(p['lam_grad_max'])) > 1e-5):
            raise RuntimeError(f'Saved HF reference mismatch: {key}')
        mp8 = lf_mp.lf_mp2_unrestricted_reference_point(
            tmat, U, g, omega, NELEC, coeff, refs['mo_energy'][i], lam0,
            np.zeros(L), max_total=8, max_excited_modes=2,
        )['total_energy']
        mp12 = lf_mp.lf_mp2_unrestricted_reference_point(
            tmat, U, g, omega, NELEC, coeff, refs['mo_energy'][i], lam0,
            np.zeros(L), max_total=12, max_excited_modes=L,
        )['total_energy']
        if abs(mp8-float(p['lf_mp2_energy'])) > 1e-8:
            raise RuntimeError(f'Saved MP2 mismatch: {key}')
        orbit_lam, orbit_shift = lf_mr.lf_translation_orbit(lam0, np.zeros(L))
        one, _, _ = measure(tmat, g, omega, orbit_lam[:1], orbit_shift[:1], occ)
        orbit, _, _ = measure(tmat, g, omega, orbit_lam, orbit_shift, occ)
        lam = np.concatenate((orbit_lam, np.zeros((1, L, L))))
        shift = np.concatenate((orbit_shift, np.full((1, L), -g/omega*sum(NELEC)/L)))
        mr, c, rho_mr = measure(tmat, g, omega, lam, shift, occ)
        if not (mr['energy'] <= orbit['energy']+1e-8 <= one['energy']+2e-8 <= hf+3e-8):
            raise RuntimeError(f'Variational nesting failed: {key}')
        e_ed = float(ed['energy'])
        if mr['energy'] < e_ed-float(ed['cutoff_tolerance'])-1e-8:
            raise RuntimeError(f'MR below converged ED beyond tolerance: {key}')
        row = dict(omega=omega, alpha=alpha, g=g, L=L, neleca=2, nelecb=2, U=U, t=T,
                   convention='paper_uncentered', frame_recipe=RECIPE, overlap_cut=CUT,
                   hf_energy=hf, mp2_energy=mp8, mp2_energy_max12_all_modes=mp12,
                   mp2_sum_change=mp12-mp8, ed_energy=e_ed,
                   ed_Nmax=int(ed['Nmax']), ed_dimension=int(ed['D']),
                   ed_residual=float(ed['residual']), ed_delta_E=ed['delta_E'],
                   ed_energy_centered=float(ed['energy_centered']),
                   ed_paper_shift=float(ed['paper_energy_shift']),
                   ed_cutoff_converged=ed['cutoff_converged'],
                   ed_cutoff_tolerance=float(ed['cutoff_tolerance']),
                   mr_one_frame_energy=one['energy'], mr_orbit_energy=orbit['energy'],
                   hf_start=p['selected_start'])
        row.update({f'mr_{name}': value for name, value in mr.items()})
        for name, energy in (('hf', hf), ('mp2', mp8), ('mr', mr['energy'])):
            row[f'{name}_minus_ed'] = energy-e_ed
        for name, density in (('hf', rho_hf), ('mr', rho_mr)):
            row[f'{name}_charge_imbalance'] = float(np.ptp(density.sum(axis=0)))
            row[f'{name}_spin_amplitude'] = float(np.max(np.abs(density[0]-density[1])))
            for spin in range(2):
                for site in range(L):
                    row[f'{name}_rho_{spin}_{site}'] = density[spin, site]
        row['mr_density_near_degenerate'] = mr['gap'] < 1e-8
        row['mr_closer_than_mp2'] = abs(row['mr_minus_ed']) < abs(row['mp2_minus_ed'])
        tag = f'point_{len(records):02d}'
        states.update({tag+'_lam': lam, tag+'_shift': shift, tag+'_coeff': c})
        records.append(row)
        print(f'{omega=:g} {alpha=:g} errors HF={row["hf_minus_ed"]:.6g} '
              f'MP2={row["mp2_minus_ed"]:.6g} MR={row["mr_minus_ed"]:.6g} '
              f'rank={mr["rank"]} gap={mr["gap"]:.3g}', flush=True)
    output = ROOT/'data/fig2d_mr.csv'
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(ROOT/'data/fig2d_mr_states.npz',
                        omega=[r['omega'] for r in records],
                        alpha=[r['alpha'] for r in records], **states)
    sources = ['data/fig2d_exact.csv', 'data/fig2d_lf.csv', 'data/fig2d_lf_references.npz',
               'src/lf_mr.py', 'src/lf_mp.py', 'scripts/fig2d_mr_scan.py']
    metadata = dict(frame_recipe=RECIPE, source_sha256={
        f: hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources},
        ed_recomputed=False, ed_validation='Stored converged ED; parameters, shift and residual audited',
        mp2_plotted='Original max_total=8, max_excited_modes=2; recomputed and checked against 12/all modes',
        mr_frame_converged=False,
        mr_frame_note='Fixed five frames, no optimization of MR energy; one-frame/orbit comparisons recorded',
        density_note='Raw lowest eigenvector density; gap <1e-8 flagged as basis dependent',
        density_formulas='w_ij=sum_A C_Aij*(S C)_Aij; rho_alpha=sum_ij w_ij occ_alpha_ip; beta analog',
        max_mp2_sum_change=max(abs(r['mp2_sum_change']) for r in records),
        max_projected_residual=max(r['mr_residual_projected'] for r in records),
        max_overlap_cutoff_spread=max(r['mr_cutoff_energy_spread'] for r in records))
    (ROOT/'data/fig2d_mr_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return records


def plot(records, focus=False):
    omegas = (0.5,) if focus else (0.5, 5.0)
    fig, axes = plt.subplots(3, len(omegas), figsize=(6.4*len(omegas), 9.0),
                             squeeze=False, sharex='col', constrained_layout=True,
                             gridspec_kw={'height_ratios': (2.3, 2, 1.6)})
    colors = {'ed': 'tab:purple', 'hf': 'tab:orange', 'mp2': 'tab:red', 'mr': 'tab:brown'}
    labels = {'ed': 'Exact ED', 'hf': 'LF-HF', 'mp2': 'LF-MP2', 'mr': 'LF-MR (fixed frames)'}
    for col, omega in enumerate(omegas):
        rows = [r for r in records if r['omega']==omega and (not focus or r['alpha']<=2.0)]
        x = np.array([r['alpha'] for r in rows])
        for name, marker in (('ed','s'), ('hf','o'), ('mp2','D'), ('mr','v')):
            axes[0,col].plot(x, [r[name+'_energy'] for r in rows], label=labels[name],
                             color=colors[name], marker=marker, markersize=4, linewidth=1.4)
            if name!='ed':
                axes[1,col].plot(x, [r[name+'_minus_ed'] for r in rows], label=labels[name],
                                 color=colors[name], marker=marker, markersize=4, linewidth=1.4)
        axes[0,col].set_title(f'$\\omega={omega:g}$')
        axes[0,col].set_ylabel('$E/|t|$')
        axes[0,col].legend(frameon=False, fontsize=8, ncol=2)
        axes[1,col].axhline(0, color='black', linewidth=.7)
        axes[1,col].set_ylabel('Signed error $(E-E_{ED})/|t|$')
        if not focus:
            axes[1,col].set_yscale('symlog', linthresh=1e-3)
        for name in ('hf','mr'):
            for field, style, suffix in (('spin_amplitude','-','spin'),('charge_imbalance','--','charge')):
                vals=np.array([r[name+'_'+field] for r in rows])
                ambiguous=np.array([r['mr_density_near_degenerate'] for r in rows],dtype=bool)
                if name=='mr' and field=='charge_imbalance':
                    axes[2,col].plot(x[ambiguous], vals[ambiguous], 'x', color='gray',
                                     label='MR near-degenerate root' if ambiguous.any() else None)
                    vals[ambiguous]=np.nan
                axes[2,col].plot(x, vals, style, color=colors[name], marker='o' if name=='hf' else 'v',
                                 markersize=3, label=f'{name.upper()} {suffix}')
        axes[2,col].set_ylabel('Spin / charge imbalance')
        axes[2,col].set_xlabel('$\\alpha=g^2/\\omega$')
        axes[2,col].legend(frameon=False, fontsize=7.5, ncol=2)
        for ax in axes[:,col]:
            ax.grid(alpha=.2)
    fig.suptitle('Four-site Hubbard-Holstein: $N_e=4$, $U=4$, $t=-1$\n'
                 'LF-MR: translated HF frame + uniform CS frame\n'
                 'Full electronic CI per frame', fontsize=12)
    fig.supxlabel('Spin: max |rho_alpha - rho_beta|; charge: max n - min n. '
                  'ED reused with verified cutoff metadata.', fontsize=8)
    stem='fig2d_mr_afm' if focus else 'fig2d_mr'
    for ext in ('png','pdf'):
        fig.savefig(ROOT/f'figures/{stem}.{ext}', dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    lib.num_threads(1)
    rows = run_scan()
    plot(rows)
    plot(rows, focus=True)
