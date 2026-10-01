"""Direct relative-translation kernels versus explicit orbit projection."""
import numpy as np
import pytest
from src import lf_mr as mr


@pytest.mark.parametrize('L,nelec,q',[(3,(1,0),1),(4,(2,2),1),(4,(2,2),-1),
                                    (3,(2,1),1),(2,(0,0),1),(4,(4,1),-1)])
@pytest.mark.parametrize('fixed',[False,True])
def test_direct_matches_explicit_projection(L,nelec,q,fixed):
    rng=np.random.default_rng(93)
    lam=rng.normal(size=(2,L,L))*.25
    z=rng.normal(size=(2,L))*.2
    t=-(np.roll(np.eye(L),1,axis=0)+np.roll(np.eye(L),-1,axis=0))+.3*np.eye(L)
    from math import comb
    ci=rng.normal(size=(2,comb(L,nelec[0]),comb(L,nelec[1]))) if fixed else None
    args=(t,3.,.7,.8,lam,z,nelec)
    full=mr.lf_translation_projected_noci(*args,character=q,electronic_coeff=ci,backend='full',return_projection=True)
    direct=mr.lf_translation_projected_noci(*args,character=q,electronic_coeff=ci)
    assert direct['projection'] is None
    for key in ('hmat','smat'):
        np.testing.assert_allclose(direct[key],full[key],atol=2e-13,rtol=1e-13)
    assert direct['energy']==pytest.approx(full['energy'],abs=2e-10)
    assert direct['rank']==full['rank']
    assert direct['residual_projected']<1e-9
    expected=full['projection']@direct['coeff'].ravel()
    np.testing.assert_allclose(direct['expanded_coeff'].ravel(),expected,atol=1e-13)


def test_direct_does_not_build_full_orbit_kernels(monkeypatch):
    old=mr.lf_multi_frame_hamiltonian
    def checked(t,U,g,w,lam,z,n):
        assert len(lam)==1  # Only the existing single-seed validation call.
        return old(t,U,g,w,lam,z,n)
    monkeypatch.setattr(mr,'lf_multi_frame_hamiltonian',checked)
    rng=np.random.default_rng(5)
    lam=rng.normal(size=(3,4,4))*.2
    t=-(np.roll(np.eye(4),1,axis=0)+np.roll(np.eye(4),-1,axis=0))
    out=mr.lf_translation_projected_noci(t,4.,.3,1.,lam,np.zeros((3,4)),(2,2))
    assert out['hmat'].shape==(108,108)
