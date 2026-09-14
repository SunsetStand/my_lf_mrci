import numpy as np

from src import my_direct_ep as myep

if __name__ == "__main__":
    nmax_values = range(11)
    L = 4
    nelec = (1,0)
    omega = 0.5
    alpha = 1.0
    U = 0
    t = -1
    g = myep.alpha_to_g(alpha, omega)
    Ht = myep.electron_ring_hopping(L, t)
    hpp = np.eye(L) * omega

    print("coupling_convention = paper_uncentered")
    print(f"L={L}, nelec={nelec}, omega={omega}, alpha={alpha}, g={g}")

    print("Nmax\tD\tEnergy\tResidual\tdelta_E")
    E_prev = None
    for nmax in nmax_values:
        energy, _, residual = myep.kernel(Ht,U,g,hpp,L,nelec,nmax)
        D = int(np.prod(myep.make_shape(L, nelec, nmax)))
        if E_prev is None:
            print(f"{nmax}\t{D}\t{energy:.12f}\t{residual:.2e}\t{'N/A'}")
        else:
            delta_E = abs(energy - E_prev)
            print(f"{nmax}\t{D}\t{energy:.12f}\t{residual:.2e}\t{delta_E:.2e}")
        E_prev = energy
    
