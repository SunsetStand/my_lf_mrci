# Week 1

Reproduced the exact-diagonalization (ED) results, using PySCF's `fci.direct_ep` implementation as a reference.

# Week 2

1. Found that a large phonon cutoff was required when the phonon coordinates were not centered. I therefore implemented a centered formulation, which improved cutoff convergence in the cases tested. I also reproduced many-electron benchmark results and checked agreement with PySCF.
2. Reproduced the CS-HF, CS-MP2, and LF-HF calculations and observed the symmetry breaking in LF-HF reported in the paper. I also found that CS-HF could converge to a symmetry-broken solution when initialized with symmetry-broken orbitals.

# Week 3

1. Initially, my LF-HF energies were slightly higher than those in the paper. With AI assistance, I compared the paper and its associated code with my implementation and found that the paper used a full parameter matrix $\lambda_{xp}$, whereas I had restricted it to a diagonal form with fewer variational parameters. For the one-electron ansatz, the physical state depends on the conditional displacements $z_x-\lambda_{xp}$. I removed this parameter redundancy by choosing the gauge $z_x=0$ and optimizing the full matrix $\lambda_{xp}$, obtaining energies consistent with the published curve.
2. Implemented LF-MP2 and reproduced the published results. It substantially improves the weak-coupling energies, but provides only limited corrections in the symmetry-broken region of the scan. This motivated a multireference treatment.
3. Developed a multi-frame LF variational calculation. Starting from LF-HF reference frames, I included their translations over all four sites to allow restoration of translation symmetry. Each frame includes all four electronic site states, and the expansion coefficients are optimized by solving the nonorthogonal CI equation
   $$
   H C = S C \varepsilon.
   $$
   The full-$\alpha$ scan uses eight frames: the translation orbits of the lowest-energy LF-HF solution found and the LF-HF solution obtained from a symmetric coherent-state initial guess. At $\alpha=2.4$, this reduces the energy error relative to ED from about $0.1092$ to $0.00494$. A separate calculation adds translation orbits of interpolated and extrapolated LF frames; with 32 frames, the error decreases to about $0.00101$, although the final energies remain sensitive to the overlap cutoff. At smaller $\alpha$, the selected frames do not enlarge the effective variational space, so the energy remains equal to LF-HF. A perturbative correction to MR-LF is a possible next step, alongside improving the reference-frame space.
