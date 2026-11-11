argon_clusters.xyz holds three 27-atom argon clusters. I want all three relaxed together
to a maximum force below 0.05 eV/A, with at most 200 optimizer steps, using a
Lennard-Jones potential (epsilon 0.0104 eV, sigma 3.4 A, cutoff 8.5 A). Later I will run
thousands of these on a GPU, so treat them as one batch on the device and use the
batched atomistic simulation library installed in this environment, not a loop of ASE
optimizers. This machine has no GPU, so check it here on CPU, but write it so it uses a
GPU when one is there. Print the final energy and maximum force of each cluster. When
you are done, tell me the name of the script you wrote and which library classes and
functions it uses.
