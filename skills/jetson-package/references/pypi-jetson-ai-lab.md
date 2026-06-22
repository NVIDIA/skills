# Jetson AI Lab devpi (PyPI indexes)

Root URL: [https://pypi.jetson-ai-lab.io/](https://pypi.jetson-ai-lab.io/)

The site is a **devpi** mirror with **per–JetPack / CUDA** subtrees (for example `jp6/cu126`, `jp6/cu128`). Paths evolve with JetPack lines — **open the root in a browser** and pick the tree that matches:

- Your **L4T / JetPack** line (`cat /etc/nv_tegra_release`)
- The **CUDA userland** on the device (see JetPack release notes)

## pip usage pattern

After you identify the correct index URL from the devpi tree:

```bash
pip install --extra-index-url '<paste-index-url-here>' '<package>'
```

Or set `PIP_EXTRA_INDEX_URL` for a session. Prefer **Jetson AI Lab indexes** for GPU-heavy wheels over plain PyPI when on Jetson.

## Why not only PyPI?

Public `manylinux2014_aarch64` wheels may lack **SM 8.7 (Orin)** or **SM 11.0 (Thor)** native code paths. The Jetson AI Lab indexes publish builds intended for Jetson GPUs.
