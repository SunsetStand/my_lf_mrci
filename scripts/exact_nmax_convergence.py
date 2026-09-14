import numpy as np

from src import my_direct_ep as myep


def run_convergence(alpha: float, nmax_values, cutoff_tolerance: float = 1e-6, required_consecutive: int = 2):
    L = 4
    nelec = (1,0)
    omega = 0.5
    U = 0
    t = -1
    g = myep.alpha_to_g(alpha, omega)
    Ht = myep.electron_ring_hopping(L, t)
    hpp = np.eye(L) * omega
    records = []
    cutoff_converged = False

    print("coupling_convention = paper_uncentered")
    print(f"L={L}, nelec={nelec}, omega={omega}, alpha={alpha}, g={g}")

    print("Nmax\tD\tEnergy\tResidual\tdelta_E")
    E_prev = None
    consecutive_converged_count = 0
    for nmax in nmax_values:
        energy, _, residual = myep.kernel(Ht,U,g,hpp,L,nelec,nmax)
        D = int(np.prod(myep.make_shape(L, nelec, nmax)))
        delta_E = abs(energy - E_prev) if E_prev is not None else None
        delta_text = "N/A" if delta_E is None else f"{delta_E:.2e}"
        print(f"{nmax}\t{D}\t{energy:.12f}\t{residual:.2e}\t{delta_text}")
        records.append((nmax, D, energy, residual, delta_E))
        E_prev = energy
        if delta_E is not None and delta_E < cutoff_tolerance:
            consecutive_converged_count += 1
        else:
            consecutive_converged_count = 0
        if consecutive_converged_count >= required_consecutive:
            cutoff_converged = True
            break

    return records, cutoff_converged


if __name__ == "__main__":
    records, cutoff_converged = run_convergence(alpha=3.0, nmax_values=range(16,27,2), cutoff_tolerance=1e-6, required_consecutive=2)
    print(f"Converged: {cutoff_converged} (delta_E={records[-1][4]:.2e})")
    selected_Nmax = records[-1][0] if cutoff_converged else None
    print(f"Selected Nmax: {selected_Nmax}")
    selected_energy = records[-1][2] if cutoff_converged else None
    print(f"Selected Energy: {selected_energy:.12f}" if selected_energy is not None else "Selected Energy: N/A")
