I have a small message-passing-style network in input/toy_pair_model.py (class
TinyPairModel). It takes atomic numbers, positions, and a batch index tensor, and
returns one scalar energy per graph. Wrap it so toolkit dynamics stages (FIRE2, NVT) can
call it: energy plus forces derived from the energy by autograd. Then run a quick sanity
check with a single 4-atom structure and print the energy and force shapes it produces.
This machine has no GPU, so check it here on CPU at this size, but write it so it uses a
GPU when one is there. When you are done, tell me the name of the script you wrote and
which toolkit classes and functions it uses.
