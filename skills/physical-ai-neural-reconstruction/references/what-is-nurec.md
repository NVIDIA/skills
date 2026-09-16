# What is NuRec?

**NuRec** (NVIDIA Omniverse Neural Reconstruction) takes camera,
LiDAR, radar, or stereo recordings — typically from a self-driving car
or a robot — and turns them into a 3D scene you can re-render from any
viewpoint. Names that come up a lot:

- **NRE** — "Neural Reconstruction Engine". NuRec is the product; NRE
  is the engine that trains and renders. Both route to the upstream
  `nre` skill.
- **USDZ** — the file format of a trained scene. A zip archive that
  Omniverse, Isaac Sim, and CARLA can open.
- **NCore V4** — the input format NRE consumes. Raw recordings must be
  converted to NCore V4 before training.
- **3DGUT / 3DGRT** — the two 3D Gaussian Splatting flavours used
  internally by NRE. The default Hydra recipe picks one; most users
  never set it manually.

A typical NuRec project has three stages:

1. **Get the input** — convert your own recording to NCore V4
   (`ncore`), or download a pre-converted dataset
   (`physical-ai-datasets`).
2. **Train the reconstruction** — feed NCore V4 to NRE; out comes a
   USDZ (`nre`).
3. **Render new views** — render images, videos, or LiDAR sweeps from
   the USDZ (`nre`).

Projects that just want to *use* an existing NVIDIA-published scene
skip step 2.
