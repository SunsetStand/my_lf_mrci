import numpy as np
import pytest
from pyscf import fci
from src import lf_mr as mr, my_direct_ep as ep


@pytest.mark.parametrize('L,nelec', [(3,(1,0)),(4,(2,2)),(3,(2,1)),(2,(0,0)),(4,(4,1))])
def test_translation_signs_against_slater_minors(L, nelec):
    rng=np.random.default_rng(24)
    lam=rng.normal(size=(1,L,L))*.1
    out=mr.lf_translation_projected_noci(-(np.roll(np.eye(L),1,axis=0)+np.roll(np.eye(L),-1,axis=0)), 2., .3, 1.,
                                        lam, np.zeros((1,L)), nelec, return_projection=True)
    strings=[ep.make_electron_basis(L,n)[0] for n in nelec]
    occupied=[[np.flatnonzero((int(s)>>np.arange(L))&1) for s in ss] for ss in strings]
    d=len(strings[0])*len(strings[1])
    for R in range(L):
        perm=np.roll(np.eye(L), R, axis=0)
        spin=[np.array([[np.linalg.det(perm[np.ix_(i,j)]) for j in occ] for i in occ]) for occ in occupied]
        expected=np.kron(*spin)/np.sqrt(L)
        np.testing.assert_allclose(out['projection'][R*d:(R+1)*d],expected,atol=1e-14)
    assert out['norm_error']<1e-10
    assert out['residual_projected']<1e-9


def test_hubbard_ground_sector_and_full_fci():
    t=ep.electron_ring_hopping(4,-1.)
    args=(t,4.,0.,.5,np.zeros((1,4,4)),np.zeros((1,4)),(2,2))
    plus=mr.lf_translation_projected_noci(*args)
    minus=mr.lf_translation_projected_noci(*args,character=-1)
    eri=np.zeros((4,)*4)
    for i in range(4): eri[i,i,i,i]=4.
    solver=fci.direct_spin1.FCI()
    e,_=solver.kernel(t,eri,4,(2,2))
    assert solver.converged
    assert minus['energy']==pytest.approx(e,abs=1e-10)
    assert plus['energy']>e+.1


def test_fixed_state_and_full_frame_variational_relation():
    L=3
    rng=np.random.default_rng(12)
    lam=rng.normal(size=(2,L,L))*.2
    z=np.zeros((2,L))
    args=(ep.electron_ring_hopping(L,-1.),0.,.4,1.,lam,z,(1,0))
    full=mr.lf_translation_projected_noci(*args)
    ci=rng.normal(size=(2,L,1)).astype(np.float64)
    fixed=mr.lf_translation_projected_noci(*args,electronic_coeff=ci)
    assert full['energy'] <= fixed['energy']+1e-12
    assert fixed['coeff'].shape==(2,)
    assert full['coeff'].shape==(2,L,1)
    with pytest.raises(ValueError): mr.lf_translation_projected_noci(*args,character=-1)


def test_single_electron_existing_two_state_regression():
    from src import lf_mp
    t=ep.electron_ring_hopping(4,-1.)
    settings=dict(nrandom=8,random_scale=.5,seed=0,gtol=1e-8,max_cycle=500)
    uniform,_=lf_mp.lf_hf_full_alpha_point(2.2,t,.5,**settings)
    local,_=lf_mp.lf_hf_full_alpha_point(2.4,t,.5,**settings)
    result=mr.lf_translation_projected_noci(
        t,0.,np.sqrt(2.2*.5),.5,np.stack((uniform.lam,local.lam)),
        np.stack((uniform.shift,local.shift)),(1,0),
        electronic_coeff=np.stack((uniform.coeff,local.coeff))[:,:,None])
    assert result['energy']==pytest.approx(-2.930178129612,abs=1e-10)


def test_projected_density_uniform_and_variational_bound():
    rng=np.random.default_rng(57)
    lam=rng.normal(size=(2,4,4))*.2
    shift=rng.normal(size=(2,4))*.1
    args=(ep.electron_ring_hopping(4,-1.),4.,.7,.5,lam,shift,(2,2))
    for q in (1,-1):
        out=mr.lf_translation_projected_noci(*args,character=q)
        S=mr.lf_multi_frame_overlap(out['lam'],out['shift'],(2,2))
        H=mr.lf_multi_frame_hamiltonian(*args[:4],out['lam'],out['shift'],(2,2))
        unrestricted,_,_=mr.lf_noci_lowest(H,S)
        assert out['energy']>=unrestricted-1e-10
        c=out['expanded_coeff']
        w=np.sum(c*(S.reshape(c.size,c.size)@c.ravel()).reshape(c.shape),axis=0)
        ss=ep.make_electron_basis(4,2)[0]
        occ=((ss[:,None]>>np.arange(4))&1)
        np.testing.assert_allclose(w.sum(0)@occ,.5,atol=1e-10)
        np.testing.assert_allclose(w.sum(1)@occ,.5,atol=1e-10)
