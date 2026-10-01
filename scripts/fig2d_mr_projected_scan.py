"""Translation-projected interpolation diagnostic, with full determinant CI.

Path: converged CS-start LF-HF seed at target -> saved alpha=4 local seed
(same omega), linearly interpolating lambda and z without rescaling.
All seeds are projected separately before NOCI. Both real momentum sectors
are reported; the equal-weight (+1) sector is never silently substituted.
"""
import csv
import hashlib
import json
import numpy as np
from pyscf import lib
from scripts import fig2d_mr_scan as b

ROOT=b.ROOT


def solve(lam,z,t,g,w,q,ci=None):
    out=b.lf_mr.lf_translation_projected_noci(t,4.,g,w,lam,z,(2,2),
                                            character=q,electronic_coeff=ci)
    h,s=out['hmat'],out['smat']
    M=len(h)
    cuts=[]
    for cut in (1e-8,1e-12):
        cuts.append(b.lf_mr.lf_noci_lowest(h.reshape(M,1,M,1),s.reshape(M,1,M,1),overlap_cut=cut)[0])
    out['energy_cut_loose'],out['energy_cut_tight']=cuts
    out['cutoff_spread']=float(np.ptp([out['energy']]+cuts))
    c=out['expanded_coeff']
    S=b.lf_mr.lf_multi_frame_overlap(out['lam'],out['shift'],(2,2))
    weights=np.sum(c*(S.reshape(c.size,c.size)@c.ravel()).reshape(c.shape),axis=0)
    strings=b.ep.make_electron_basis(4,2)[0]
    occ=((strings[:,None]>>np.arange(4))&1)
    rho=np.stack((weights.sum(1)@occ,weights.sum(0)@occ))
    out['charge_imbalance']=float(np.ptp(rho.sum(0)))
    out['spin_amplitude']=float(np.max(np.abs(rho[0]-rho[1])))
    out['rho']=rho
    out['numerically_stable']=bool(out['cutoff_spread']<1e-7 and out['norm_error']<1e-8
                                  and out['residual_projected']<1e-8)
    if not np.allclose(rho.sum(1),2.,atol=1e-8,rtol=0) or out['charge_imbalance']>1e-8:
        raise RuntimeError('Projected density failed translation/normalization check')
    return out


def slater_ci(coeff):
    strings=b.ep.make_electron_basis(4,2)[0]
    occupied=np.stack([np.flatnonzero((int(s)>>np.arange(4))&1) for s in strings])
    minors=[np.linalg.det(coeff[spin,occupied,:2]) for spin in range(2)]
    return np.outer(*minors)


def run():
    exact,_,refs,ix=b.read_sources()
    original=b._read(ROOT/'data/fig2d_mr.csv')
    t=b.ep.electron_ring_hopping(4,-1.)
    records, branches, states=[],[],{}
    for w,a in sorted(original):
        g=np.sqrt(a*w)
        c0=refs['mo_coeff'][ix[w,0.]]
        cs=b.lf_mp.lf_hf_multi_optimize(t,4.,g,w,(2,2),c0[0],c0[1],
                                      np.full((4,4),g/(4*w)),gtol=1e-6,max_cycle=1000)
        branches.append(dict(omega=w,alpha=a,success=bool(cs.success),energy=float(cs.fun),
                             orbital_grad_norm=float(cs.orbital_grad_norm),
                             lam_grad_max=float(cs.lam_grad_max),
                             charge_imbalance=float(np.ptp(cs.spin_density.sum(0))),
                             spin_amplitude=float(np.max(np.abs(cs.spin_density[0]-cs.spin_density[1]))),
                             message=str(cs.message)))
        seed_tag=f'cs_{w:g}_{a:g}'
        states.update({seed_tag+'_lam':cs.lam,seed_tag+'_mo_coeff':cs.mo_coeff,
                       seed_tag+'_shift':cs.shift,seed_tag+'_rho':cs.spin_density})
        print('CS-start',w,a,cs.success,cs.fun,flush=True)
        if not cs.success:
            continue
        best=refs['lam'][ix[w,a]]
        local=refs['lam'][ix[w,4.]]
        fixed_ci=np.stack((slater_ci(cs.mo_coeff),slater_ci(refs['mo_coeff'][ix[w,4.]])))
        last={}
        for stage,npath in [('fixed_two_states',2),('original_projected',0),
                            ('endpoints',2),('grid3',3),('grid5',5),('grid9',9)]:
            theta=np.linspace(0,1,npath) if npath else np.array([])
            if stage=='fixed_two_states':
                lam=np.stack((cs.lam,local))
                z=np.stack((cs.shift,np.zeros(4)))
                ci=fixed_ci
            else:
                # Preserve the complete original five-frame span within each sector.
                lam=np.concatenate((best[None],np.zeros((1,4,4)),
                                    (1-theta[:,None,None])*cs.lam+theta[:,None,None]*local))
                z=np.concatenate((np.zeros((1,4)),np.full((1,4),-g/w),
                                  (1-theta[:,None])*cs.shift))
                ci=None
            for q in (1,-1):
                try:
                    out=solve(lam,z,t,g,w,q,ci)
                except ValueError as exc:
                    # A whole-state momentum projection may be identically zero.
                    if stage=='fixed_two_states' and 'No eigenvalues' in str(exc):
                        print('Null whole-state sector',w,a,q,flush=True)
                        continue
                    raise
                E=out['energy']
                if stage!='fixed_two_states':
                    if q in last and E>last[q]+1e-7:
                        print('WARNING cutoff-dependent nesting',w,a,stage,q,E-last[q],flush=True)
                    last[q]=E
                row=dict(omega=w,alpha=a,g=g,L=4,neleca=2,nelecb=2,U=4.,t=-1.,
                         convention='paper_uncentered',stage=stage,character=q,
                         seed_alpha_local=4.,nseed=len(lam),theta=json.dumps(theta.tolist()),
                         energy=E,ed_energy=float(exact[w,a]['energy']),
                         minus_ed=E-float(exact[w,a]['energy']),
                         original_energy=float(original[w,a]['mr_energy']),
                         mp2_energy=float(original[w,a]['mp2_energy']))
                for field in ('rank','norm_error','residual_projected','cutoff_spread',
                              'numerically_stable','charge_imbalance','spin_amplitude',
                              'energy_cut_loose','energy_cut_tight'):
                    row[field]=out[field]
                if E<row['ed_energy']-float(exact[w,a]['cutoff_tolerance'])-1e-8:
                    raise RuntimeError('Below ED beyond cutoff tolerance')
                tag=f'point_{len(records):03d}'
                states.update({tag+'_lam':lam,tag+'_shift':z,tag+'_coeff':out['coeff'],
                               tag+'_rho':out['rho']})
                records.append(row)
                print(w,a,stage,q,'error',row['minus_ed'],'rank',out['rank'],
                      'cut',out['cutoff_spread'],flush=True)
    for name,rows in [('fig2d_mr_projected',records),('fig2d_mr_cs_start',branches)]:
        with (ROOT/f'data/{name}.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
            wr.writeheader()
            wr.writerows(rows)
    np.savez_compressed(ROOT/'data/fig2d_mr_projected_states.npz',**states)
    sources=['src/lf_mr.py','scripts/fig2d_mr_projected_scan.py','data/fig2d_mr.csv',
             'data/fig2d_exact.csv','data/fig2d_lf_references.npz']
    meta=dict(source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              cs_start='converged g=0 UHF orbitals (uniform charge, staggered spin); uniform CS phonons',
              cs_start_failures=[r for r in branches if not r['success']],
              nonlinear_mr_optimized=False,frame_converged=False,ed_recomputed=False,
              sectors='q=+1 equal weight; q=-1 alternating sign, both include fermionic signs',
              path='target CS-start LF-HF lambda,z to same-omega alpha=4 lambda,z; raw linear interpolation',
              max_cutoff_spread=max(r['cutoff_spread'] for r in records))
    (ROOT/'data/fig2d_mr_projected_metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    plot(records)


def plot(rows):
    plt=b.plt
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex='col',constrained_layout=True)
    for j,w in enumerate((.5,5.)):
        for stage,color in [('original_projected','tab:brown'),('endpoints','tab:blue'),
                            ('grid3','tab:cyan'),('grid5','tab:orange'),('grid9','tab:green')]:
            pairs={a:{r['character']:r for r in rows if r['omega']==w and r['alpha']==a and r['stage']==stage}
                   for a in sorted({r['alpha'] for r in rows if r['omega']==w})}
            x=list(pairs)
            axes[0,j].plot(x,[pairs[a][1]['minus_ed'] for a in x],'.-',color=color,label=stage)
            axes[1,j].plot(x,[min(pairs[a][q]['minus_ed'] for q in (1,-1)) for a in x],'.-',color=color,label=stage)
            for panel in (0,1):
                chosen=[pairs[a][1] if panel==0 else min(pairs[a].values(),key=lambda r:r['energy']) for a in x]
                bad=[r for r in chosen if not r['numerically_stable']]
                axes[panel,j].plot([r['alpha'] for r in bad],[r['minus_ed'] for r in bad],
                                  'x',color='magenta',markersize=7,markeredgewidth=1.5,linestyle='none')
        ref=[r for r in rows if r['omega']==w and r['stage']=='original_projected' and r['character']==1]
        for ax in axes[:,j]:
            ax.plot([r['alpha'] for r in ref],[r['original_energy']-r['ed_energy'] for r in ref],
                    'k--',label='Original unrestricted MR',linewidth=1)
            ax.plot([r['alpha'] for r in ref],[abs(r['mp2_energy']-r['ed_energy']) for r in ref],
                    ':',color='tab:red',label='MP2 |error|')
            ax.set_yscale('symlog',linthresh=1e-4)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        axes[0,j].set_title(f'omega={w:g}: equal-weight q=+1')
        axes[1,j].set_title('Lower of q=+1 and q=-1')
        axes[1,j].set_xlabel('alpha = g^2 / omega')
        for ax in axes[:,j]: ax.set_ylabel('Energy error relative to ED / |t|')
    fig.suptitle('Projected LF-MR interpolation: L=4, Ne=4, U=4\n'
                 'Full electronic CI per family; magenta x: cutoff/residual check failed')
    for ext in ('png','pdf'): fig.savefig(ROOT/f'figures/fig2d_mr_projected.{ext}',dpi=200)
    plt.close(fig)


if __name__=='__main__':
    lib.num_threads(1)
    run()
