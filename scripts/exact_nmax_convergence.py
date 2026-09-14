import numpy as np

from src import my_direct_ep as myep


def run_convergence(alpha: float, nmax_values):
    L = 4
    nelec = (1,0)
    omega = 0.5
    U = 0
    t = -1
    g = myep.alpha_to_g(alpha, omega)
    Ht = myep.electron_ring_hopping(L, t)
    hpp = np.eye(L) * omega
    records = []

    print("coupling_convention = paper_uncentered")
    print(f"L={L}, nelec={nelec}, omega={omega}, alpha={alpha}, g={g}")

    print("Nmax\tD\tEnergy\tResidual\tdelta_E")
    E_prev = None
    for nmax in nmax_values:
        energy, _, residual = myep.kernel(Ht,U,g,hpp,L,nelec,nmax)
        D = int(np.prod(myep.make_shape(L, nelec, nmax)))
        delta_E = abs(energy - E_prev) if E_prev is not None else None
        delta_text = "N/A" if delta_E is None else f"{delta_E:.2e}"
        print(f"{nmax}\t{D}\t{energy:.12f}\t{residual:.2e}\t{delta_text}")
        records.append((nmax, D, energy, residual, delta_E))
        E_prev = energy

    return records


if __name__ == "__main__":
    records = run_convergence(alpha=3.0, nmax_values=range(0,17,2))
    cutoff_tolerance = 1e-6
    cutoff_converged = (records[-1][4] is not None) and (records[-1][4] < cutoff_tolerance)
    print(f"Converged: {cutoff_converged} (delta_E={records[-1][4]:.2e})")