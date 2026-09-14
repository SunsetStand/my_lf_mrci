import numpy as np

# from pyscf import lib
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

def phonon_configs(L, Nmax):
    if L <= 0 or isinstance(L, (int, np.integer)) == False:
        raise ValueError("L must be positive and an integer")
    if Nmax < 0 or isinstance(Nmax, (int, np.integer)) == False:
        raise ValueError("Nmax must be non-negative and an integer")
    d = Nmax +1
    P = d**L
    configs = np.zeros((P,L), dtype = np.int64)
    for p in range(P):
        configs[p] = decode(p,L,d)
    return configs

def boson_operators(Nmax):
    if Nmax < 0 or isinstance(Nmax, (int, np.integer)) == False:
        raise ValueError("Nmax must be non-negative and an integer")
    d = Nmax + 1
    b = np.zeros((d,d), dtype = np.float64)
    for i in range(d):
        for j in range(d):
            if i == j-1:
                b[i,j] = np.sqrt(j)
    bdag = b.conj().T
    number = bdag @ b
    return b,bdag,number

def make_electron_basis(L,N):
    if N < 0 or N > L or isinstance(N, (int,np.integer)) == False:
        raise ValueError("N must be non-negative, no greater than L and an integer")
    if L <= 0 or L>=64 or isinstance(L, (int,np.integer)) == False:
        raise ValueError("L must be positive, less than 64 and an integer")
    strings = cistring.make_strings(range(L), N)
    links = cistring.gen_linkstr_index(range(L),N,strings)
    return strings, links

def make_shape(L, nelec, Nmax):
    if L <= 0 or L>=64 or isinstance(L, (int,np.integer)) == False:
        raise ValueError("L must be positive, less than 64 and an integer")
    if isinstance(nelec,tuple) == False or len(nelec)!=2:
        raise ValueError("nelec must be a tuple of 2 elements")
    for n in nelec:
        if n<0 or n > L or isinstance(n, (int,np.integer)) == False:
            raise ValueError("nelec must be a tuple of integer between 0 and L")
    if Nmax < 0 or isinstance(Nmax, (int, np.integer)) == False:
        raise ValueError("Nmax must be non-negative and an integer")
    neleca, nelecb = nelec
    num_strings_a = cistring.num_strings(L,neleca)
    num_strings_b = cistring.num_strings(L,nelecb)
    d = Nmax + 1
    return (num_strings_a, num_strings_b) + (d,) * L

def contract_1e(tmat: np.ndarray, psi_site: np.ndarray, L: int, nelec: tuple[int,int], Nmax: int):
    psi_shape = make_shape(L,nelec,Nmax)
    if not isinstance(tmat, np.ndarray) or tmat.shape != (L, L):
        raise ValueError("tmat must be a NumPy array with shape (L, L)")

    if not isinstance(psi_site, np.ndarray) or psi_site.shape != psi_shape:
        raise ValueError("psi_site has an incompatible shape")
    new_psi = np.zeros(psi_shape)
    neleca, nelecb = nelec
    _, links_a = make_electron_basis(L, neleca)
    _, links_b = make_electron_basis(L, nelecb)
    for str0, tab in enumerate(links_a):
        for a, i, str1, sign in tab:
            new_psi[str1] += sign * tmat[a,i] * psi_site[str0]
    for str0, tab in enumerate(links_b):
        for a, i, str1, sign in tab:
            new_psi[:, str1] += sign * tmat[a,i] * psi_site[:, str0]
    return new_psi

def contract_2e_hubbard(U: float, psi_site: np.ndarray, L: int, nelec: tuple[int,int], Nmax: int):
    psi_shape = make_shape(L,nelec,Nmax)
    if not isinstance(psi_site, np.ndarray) or psi_site.shape != psi_shape:
        raise ValueError("psi_site has an incompatible shape")
    if not isinstance(U, (int, float, np.integer, np.floating)) or not np.isfinite(U):
        raise ValueError("U must be a finite real number")
    new_psi = np.zeros(psi_shape)
    neleca, nelecb = nelec
    strs_a, _ = make_electron_basis(L,neleca)
    strs_b, _ = make_electron_basis(L,nelecb)
    for ia, str_a in enumerate(strs_a):
        for ib, str_b in enumerate(strs_b):
            new_psi[ia,ib] = U * int(str_a & str_b).bit_count() * psi_site[ia,ib]
    return new_psi

def contract_pp(hpp: np.ndarray, psi_site: np.ndarray, L:int, nelec: tuple[int,int], Nmax: int):
    psi_shape = make_shape(L,nelec,Nmax)
    if not isinstance(psi_site, np.ndarray) or psi_site.shape != psi_shape:
        raise ValueError("psi_site has an incompatible shape")
    if not isinstance(hpp, np.ndarray) or not hpp.shape == (L,L):
        raise ValueError("hpp has an incompatible shape")
    if (
    not np.issubdtype(hpp.dtype, np.number)
    or np.iscomplexobj(hpp)
    or not np.all(np.isfinite(hpp))
    ):
        raise ValueError("hpp must contain finite real numbers")
    hpp_off_diagonal = hpp - np.diag(np.diag(hpp))
    if not np.allclose(hpp_off_diagonal, 0.0):
        raise ValueError("hpp must be diagonal")
    omega = np.diag(hpp)
    configs = phonon_configs(L,Nmax)
    d = Nmax + 1
    phonon_energies = configs @ omega
    energy_tensor = phonon_energies.reshape((1,1)+(d,)*L)
    new_psi = energy_tensor * psi_site
    return new_psi

def contract_ep_paper(g: float, psi_site: np.ndarray, L: int, nelec: tuple[int,int], Nmax: int):
    if isinstance(g, (bool,np.bool_)) or not isinstance(g, (int, float, np.integer, np.floating)) or not np.isfinite(g):
        raise ValueError("g must be finite real scalar")
    psi_shape = make_shape(L,nelec,Nmax)
    if psi_site.shape != psi_shape:
        raise ValueError("psi_site has the wrong shape")
    nelec_a, nelec_b = nelec
    strs_a, _ = make_electron_basis(L,nelec_a)
    strs_b, _ = make_electron_basis(L,nelec_b)
    num_a = len(strs_a)
    num_b = len(strs_b)
    occupation_site = np.zeros((num_a,num_b,L))
    for ia in range(num_a):
        for ib in range(num_b):
            for site in range(L):
                occupation_site[ia,ib,site] = bool(strs_a[ia] & (1<<site)) + bool(strs_b[ib] & (1<<site))
    b, bdag, _ = boson_operators(Nmax)
    x_local = b + bdag
    out_dtype = np.result_type(psi_site.dtype, x_local.dtype, g)
    new_psi = np.zeros(psi_shape, dtype=out_dtype)
    for site in range(L):
        occupation = occupation_site[:,:,site]
        occupation_tensor = occupation.reshape((num_a,num_b)+(1,)*L)
        weighted_psi = occupation_tensor * psi_site
        phonon_axis = 2 + site
        weighted_moved = np.moveaxis(weighted_psi, phonon_axis, -1)
        acted_moved = np.einsum('mn,...n->...m', x_local, weighted_moved)
        acted = np.moveaxis(acted_moved, -1, phonon_axis)
        new_psi += g * acted
    return new_psi

if __name__ == "__main__":
    strings, links = make_electron_basis(4,2)
    print("源地址 源占据 a i 目标地址 目标占据 sign")
    for source, string in enumerate(strings):
        source_bits = format(int(string),"04b")

        for a, i, target, sign in links[source]:
            target_bits = format(int(strings[target]),"04b")

            print(f"{source:6d} {source_bits} {a:2d} {i:2d} {target:8d} {target_bits} {sign:+d}")
