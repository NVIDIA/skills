argon_fcc32.extxyz is a 32-atom FCC argon crystal in a periodic box. Run Langevin NVT
molecular dynamics at 40 K (friction 0.01 /fs, random seed 0) for 200 steps of 2 fs,
starting from Maxwell-Boltzmann velocities at 40 K, with a Lennard-Jones potential
(epsilon 0.0104 eV, sigma 3.4 A, cutoff 8.5 A). Save a frame every 20 steps to a Zarr
store on disk so I can stream it into a training set later, then report how many frames
the store holds. The GPU-batched atomistic simulation package in this environment is
what my group uses, so build on it rather than writing an integrator or using ASE's MD.
This machine has no GPU, so check it here on CPU, but write it so it uses a GPU when one
is there. When you are done, tell me the name of the script you wrote and which library
classes and functions it uses.
