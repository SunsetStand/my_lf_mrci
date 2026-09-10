import numpy as np

# from pyscf import lib
# from pyscf import ao2mo
# from pyscf.fci import cistring
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
    if alpha < 0 or omega <= 0:
        raise ValueError("alpha should be non-negative and omega should be positive")
    return (alpha * omega)**(0.5)

def encode(occupations, d):
    if d <= 0 or isinstance(d, (int, np.integer)) == False:
        raise ValueError("d must be positive and an integer")
    for i in occupations:
        if i < 0 or i >= d or isinstance(i, (int, np.integer)) == False:
            raise ValueError("occupation must be less than d, non-negative and an integer")
    p = 0
    for i in occupations:
        p = p * d + i
    return p

def decode(p, L, d):
    if d <= 0 or isinstance(d, (int, np.integer)) == False:
        raise ValueError("d must be positive and an integer")
    if L <= 0 or isinstance(L, (int, np.integer)) == False:
        raise ValueError("L must be positive and an integer")
    if p < 0 or p >= d**L or isinstance(p, (int, np.integer)) == False:
        raise ValueError("p must be less than d**L, non-negative and an integer")
    occupations = ()
    for i in range(L):
        occupations = (p%d,) + occupations
        p = p // d
    return occupations


def electron_ring_hopping(L, t = -1.0):
    if L < 3 or isinstance(L, (int,np.integer)) == False:
        raise ValueError("L must be an integer greater than 2")
    Ht = np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            if i == (j-1)%L or i == (j+1)%L:
                Ht[i,j] = t
    return Ht

