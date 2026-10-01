"""Add alpha=0/4 HF translation orbits without rescaling their LF parameters.

Each target uses its own physical Hamiltonian and reoptimizes all CI coefficients.
The baseline is retained exactly; no core model or nonlinear optimizer is changed.
"""
import csv
import hashlib
import json

import numpy as np
from pyscf import lib
from scripts import fig2d_mr_scan as base

ROOT = base.ROOT
VARIANTS = {'low': (0.0,), 'high': (4.0,), 'both': (0.0, 4.0),
            'low_nonzero': (0.4,), 'both_nonzero': (0.4, 4.0)}


def run():
    exact, _, refs, indices = base.read_sources()
    baseline = base._read(ROOT/'data/fig2d_mr.csv')
    tmat = base.ep.electron_ring_hopping(base.L, base.T)
    strings, _ = base.ep.make_electron_basis(base.L, base.NELEC[0])
    occ = ((strings[:, None] >> np.arange(base.L)) & 1).astype(np.float64)
    records, states = [], {}
    for key in sorted(baseline):
        omega, alpha = key
        g = np.sqrt(alpha*omega)
        lam0 = refs['lam'][indices[key]]
        orbit_lam, orbit_z = base.lf_mr.lf_translation_orbit(lam0, np.zeros(base.L))
        baseline_lam = np.concatenate((orbit_lam, np.zeros((1, base.L, base.L))))
        baseline_z = np.concatenate((orbit_z, np.full((1, base.L), -g/omega)))
        check, _, _ = base.measure(tmat, g, omega, baseline_lam, baseline_z, occ)
        if abs(check['energy']-float(baseline[key]['mr_energy'])) > 1e-10:
            raise RuntimeError(f'Baseline mismatch at {key}')
        for variant, seeds in VARIANTS.items():
            frames = [(baseline_lam, baseline_z)]
            for seed in seeds:
                frames.append(base.lf_mr.lf_translation_orbit(
                    refs['lam'][indices[(omega, seed)]], np.zeros(base.L)))
            lam = np.concatenate([f[0] for f in frames])
            shift = np.concatenate([f[1] for f in frames])
            result, coeff, rho = base.measure(tmat, g, omega, lam, shift, occ)
            ed_energy = float(exact[key]['energy'])
            gain = check['energy']-result['energy']
            if gain < -1e-8:
                raise RuntimeError(f'Variational nesting failed at {key}, {variant}')
            if result['energy'] < ed_energy-float(exact[key]['cutoff_tolerance'])-1e-8:
                raise RuntimeError('Energy below ED beyond its cutoff tolerance')
            row = dict(omega=omega, alpha=alpha, g=g, variant=variant,
                       seed_alphas=json.dumps(seeds), transfer='raw_lambda_and_shift_no_rescaling',
                       L=base.L, neleca=2, nelecb=2, U=base.U, t=base.T,
                       convention='paper_uncentered', overlap_cut=base.CUT,
                       ed_energy=ed_energy, baseline_energy=check['energy'],
                       baseline_minus_ed=check['energy']-ed_energy,
                       mp2_minus_ed=float(baseline[key]['mp2_minus_ed']),
                       energy_gain=gain, minus_ed=result['energy']-ed_energy,
                       spin_amplitude=float(np.max(np.abs(rho[0]-rho[1]))),
                       charge_imbalance=float(np.ptp(rho.sum(axis=0))),
                       density_near_degenerate=result['gap'] < 1e-8, **result)
            for spin in range(2):
                for site in range(base.L):
                    row[f'rho_{spin}_{site}'] = rho[spin, site]
            tag = f'point_{len(records):03d}'
            states.update({tag+'_lam': lam, tag+'_shift': shift, tag+'_coeff': coeff})
            records.append(row)
            print(f'{omega=:g} {alpha=:g} {variant}: error={row["minus_ed"]:.9g} '
                  f'gain={gain:.9g} rank={result["rank"]}', flush=True)
    for key in baseline:
        group = {r['variant']: r for r in records if (r['omega'], r['alpha']) == key}
        for both, low in [('both', 'low'), ('both_nonzero', 'low_nonzero')]:
            if group[both]['energy'] > min(group[v]['energy'] for v in (low,'high'))+1e-8:
                raise RuntimeError(f'Combined frame nesting failed at {key}, {both}')
    with (ROOT/'data/fig2d_mr_transfer.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(ROOT/'data/fig2d_mr_transfer_states.npz',
                        omega=[r['omega'] for r in records], alpha=[r['alpha'] for r in records],
                        variant=[r['variant'] for r in records], **states)
    sources = ['data/fig2d_exact.csv', 'data/fig2d_lf_references.npz', 'data/fig2d_mr.csv',
               'src/lf_mr.py', 'scripts/fig2d_mr_scan.py', 'scripts/fig2d_mr_transfer_scan.py']
    metadata = dict(source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                    frame_recipe='baseline plus same-frequency alpha=0/4 HF orbits; alpha=0.4 control',
                    transfer='raw lambda and z; no rescaling; target Hamiltonian; full CI reoptimized',
                    nonlinear_frames_optimized=False, frame_converged=False, ed_recomputed=False,
                    max_projected_residual=max(r['residual_projected'] for r in records),
                    max_cutoff_energy_spread=max(r['cutoff_energy_spread'] for r in records),
                    min_energy_gain=min(r['energy_gain'] for r in records))
    (ROOT/'data/fig2d_mr_transfer_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    plot(records)
    return records


def plot(records):
    plt = base.plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex='col', constrained_layout=True)
    for col, omega in enumerate((0.5, 5.0)):
        reference = [r for r in records if r['omega']==omega and r['variant']=='both']
        x = [r['alpha'] for r in reference]
        axes[0,col].plot(x, [r['baseline_minus_ed'] for r in reference], 'o-',
                         color='tab:brown', label='MR baseline')
        axes[0,col].plot(x, [abs(r['mp2_minus_ed']) for r in reference], 'D:',
                         color='tab:red', label='LF-MP2 |error|')
        for variant, color, label in [('low','tab:blue','+ alpha=0'),
                                       ('high','tab:green','+ alpha=4'),
                                       ('both','black','+ alpha=0,4'),
                                       ('low_nonzero','tab:cyan','+ alpha=0.4'),
                                       ('both_nonzero','tab:purple','+ alpha=0.4,4')]:
            rows = [r for r in records if r['omega']==omega and r['variant']==variant]
            axes[0,col].plot(x, [abs(r['minus_ed']) for r in rows], '.-', color=color, label=label)
            axes[1,col].plot(x, [r['energy_gain'] for r in rows], '.-', color=color, label=label)
        axes[0,col].set_title(f'$\\omega={omega:g}$')
        axes[0,col].set_yscale('symlog', linthresh=1e-5)
        axes[0,col].set_ylabel('Absolute energy error / |t|')
        axes[0,col].legend(fontsize=8)
        axes[1,col].set_ylabel('Energy lowering from MR baseline / |t|')
        axes[1,col].set_xlabel('$\\alpha=g^2/\\omega$')
        for ax in axes[:,col]:
            ax.grid(alpha=.2)
    fig.suptitle('Four sites, four electrons, U=4, t=-1\n'
                 'Endpoint HF frames transferred unchanged; all CI coefficients reoptimized')
    for ext in ('png','pdf'):
        fig.savefig(ROOT/f'figures/fig2d_mr_transfer.{ext}', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    lib.num_threads(1)
    run()
