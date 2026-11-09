Before I wire up a real MLIP I want a quick CPU sanity check that I can drive an
equation-of-state workflow end to end with the ALCHEMI Toolkit. My structure is
si_diamond.extxyz (2x2x2 supercell of diamond silicon, 64 atoms, a = 5.43 angstrom).
Scale the lattice isotropically to 0.96, 0.98, 1.00, 1.02, and 1.04 of its current size,
relax the atoms at each volume with the cell held fixed, and fit an equation of state to
the five energy-volume points to get the equilibrium volume and bulk modulus. There is
no GPU and no trained checkpoint here, so check it on CPU, but write it so it uses a GPU
when one is there. Use the toolkit's Lennard-Jones model as a stand-in (epsilon = 0.5
eV, sigma = 2.1 angstrom, cutoff = 4.1 angstrom); I only care that the plumbing is
right, not the numbers. Cap the optimizer at 50 steps per scan point and let it stop
early on convergence with fmax = 0.05 eV/angstrom. Print the five relaxed energies and
the fitted V0 and B0. When you are done, tell me the name of the script you wrote and
which toolkit classes and functions it uses.
