import numpy as np
from pyscf import lib
# from pyscf import ao2mo
from pyscf.fci import cistring
# from pyscf.fci import rdm
# from pyscf.fci.direct_spin1 import _unpack_nelec

# source functions for electron-phonon coupling computation
# Hubbard-Houstein model
# 1 elec in 4 sites
# H = Ht + (HU) + Hph + Hep  (no double-elec term yet)

#                              site-1   ,...,site-N
#                              v             v
# ep_wfn, shape = (nstra,nstrb,nphonon+1,...,nphonon+1)
#               = (nstra,nstrb) + ([nphonon+1]*nsite)
#       For each site, {0,1,...,nphonon} gives nphonon+1 possible confs
# t for hopping, shape = (nsite,nsite)
# u for electron-electron interaction, in this model only coupled within the same site
# g for the electorn-phonon coupling
# w foe phonon frequency
# hpp for phonon-phonon interaction: (nsite,nsite)

# L = nsite
# nelec = (neleca, nelecb)
# Nmax = nphonon
# d = Nmax+1
# P = d**L
# nstra = C(L,neleca)
# nstrb = C(L,nelecb)
# S = (nstra, nstrb) + (d,)*L
# D = nstra*nstrb*P

# def  contract_all():

def alpha_to_g(alpha, omega):
    assert alpha >= 0 and omega > 0, "alpha and omega must be positive"
    return (alpha * omega)**(0.5)

def encode(occupations, d):
    assert d > 0, "d must be positive"
    for i in occupations:
        assert i >= 0, "occupations must be non-negative"
    p = 0
    for i in occupations:
        p = p * d + i
    return p

def electron_ring_hopping(L, t = -1.0):
    Ht = np.zeros((L,L))
    for i, j in range(L):
        if i == (j-1)%L or i == (j+1)%L:
            Ht[i,j] = t
    return Ht





