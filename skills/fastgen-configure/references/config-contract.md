# Config and objective contracts

Read when changing inherited recipes, geometry, data requirements or rollout policy. The
[configuration README](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/README.md) owns syntax, override grammar and
run directories; the field docstrings in [config.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/config.py) own
each setting's meaning.

## Resolution is ordered

Trace `load_config` and `setup` in [scripts.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/scripts.py) and the
helpers in [config_utils.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/config_utils.py).

- `create_config()` runs **before** CLI overrides, so values a config derives in Python
  (loader geometry from `model_pixel_shape`, frame counts, causal lengths) do not
  recompute when an override changes their inputs. After overriding latent shape,
  network variant or FPS, recheck the derived values in both the train and validation
  loaders. Loaders take `img_size=(W, H)` while pixel shapes are `(T, H, W)`.
- Copy shared presets (`.copy()`, `copy.deepcopy`) before editing them so repeated
  `create_config()` calls in one process do not leak mutations between configs.
- Hydra dict overrides merge; to remove inherited entries (callbacks, guidance terms,
  modality fields) follow the replacement procedure in `config_utils.py` and inspect the
  resulting keys, not only the values you changed.
- `instantiate` in [instantiate.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/instantiate.py) recurses through
  `_target_` entries and lists but returns a mapping without `_target_` untouched; its
  consumer instantiates such deferred entries (optimizers receive the model, for example).
- Inventory the roots a method builds: `FastGenModel.teacher_config` in
  [model.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/model.py) uses `model.teacher` when set and `model.net`
  otherwise, and auxiliary networks such as DMD2's fake score are built from it too.
  Budget every root's storage precision and optimizer state plus the preprocessors; a
  student parameter count is not a memory estimate.
- `--dryrun` resolves the config before process-group initialisation and before the
  global batch is rounded to integer accumulation steps, so read startup's logged values
  as well. The serialized `config.yaml` stringifies Python objects: it records the
  resolution but is not a re-runnable config.

## Decide supervision from the active objective

Trace the batch reads in the method rather than reasoning from its family name.

- Backward simulation changes where student inputs come from; it does not remove the
  target requirements of other enabled losses, and adversarial terms are skipped when
  `real` is absent ([prompt-only training](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx#prompt-only-training-omits-an-intended-loss)).
  A finite loss can belong to a different objective than the intended one.
- Cached KD data must match the student's anchored grid; the
  [KD documentation](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/trajectory_matching/README.md#kd-knowledge-distillation)
  states what training checks. Do not relabel arbitrary intermediate latents as paths.
- Carried rollouts are an algorithmic choice: read
  [backward_simulation.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/methods/backward_simulation.py) for `default` versus
  `staggered`/`synced`, which need [CarryCallback](https://github.com/NVlabs/FastGen/blob/main/fastgen/callbacks/carry.py) and
  constrain the world size against rollout length and update period. Keep the callback
  with the method, and check validation profiles and conditioning requirements for every
  enabled term.
