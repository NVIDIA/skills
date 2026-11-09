---
name: fastgen-runtime
description: >-
  Resolve the execution context for an NVIDIA FastGen command: checkout and interpreter, config path and override syntax, output/cache/data roots, run identity, and distributed launch variables. Use it when preparing or debugging execution; method selection belongs to fastgen-configure.
license: Apache-2.0
metadata:
  author: "Julius Berner <jberner@nvidia.com>"
  domain: ml
---

# FastGen runtime

Shared prerequisite for every FastGen command. The checkout's code and READMEs are
authoritative; this skill says where to look and what to verify. The
[configuration README](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/README.md) owns syntax, environment variables
and credentials; the [troubleshooting guide](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx) owns
failure diagnosis.

1. **Checkout and interpreter.** Locate the checkout separately from this skill directory
   and read its READMEs rather than assuming another version matches. Apply
   [fastgen-setup](https://github.com/NVlabs/FastGen/blob/main/skills/fastgen-setup/SKILL.md) only for missing work. Select the
   interpreter and devices the task actually uses and obey local scheduler, container and
   storage rules; importing GPU dependencies or resolving a config may already need an
   allocation.
2. **Config and overrides.** Run from the repository root (or put it on `PYTHONPATH`):
   training enters through the root [train.py](https://github.com/NVlabs/FastGen/blob/main/train.py).
   `--config` is a relative `.py` path exposing `create_config()`, and the loader in
   [config_utils.py](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/config_utils.py) rejects absolute paths and `..`.
   Script arguments precede the single `-` required by `override_config_with_opts`
   to introduce Hydra overrides
   ([command-line arguments](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/README.md#command-line-arguments)).
   `--dryrun` resolves the config and exits before the process group is initialised and
   before the global batch is rounded to accumulation steps; it tests neither.
3. **Roots.** Set an absolute, persistent `FASTGEN_OUTPUT_ROOT` and respect the data,
   checkpoint and Hugging Face cache roots derived from it (`.env` is loaded on import;
   see [environment variables](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/README.md#environment-variables)).
   Check the documented cache layout before pre-downloading and enable offline mode only
   once every needed file is cached. Keep credentials out of snapshots and logs.
4. **Run identity.** Independent runs need distinct `log_config.project`/`group`/`name`:
   the [run directory](https://github.com/NVlabs/FastGen/blob/main/fastgen/configs/README.md#run-directory) holds the resolved
   config, checkpoints and the W&B id, and resume is on by default.
   Also check the checkpoint destination: [resolve_save_dir](https://github.com/NVlabs/FastGen/blob/main/fastgen/utils/scripts.py)
   preserves absolute paths and S3 URIs, so those must be distinct for independent runs.
5. **Launch.** Check the entry point's distributed support before building a launcher:
   training distributes only with `trainer.ddp` or `trainer.fsdp`, and the standard
   inference scripts do not shard models. Match `RANK`, `LOCAL_RANK` and `WORLD_SIZE` to
   the processes launched; stale launcher variables make setup wait for peers
   ([distributed startup](https://github.com/NVlabs/FastGen/blob/main/fern/docs/troubleshooting.mdx#distributed-startup-fails-or-waits-for-peers)).
   Verify actual world size, wrapping, accumulation and effective batch in startup logs;
   a review-only plan should identify these checks as pending.

Report the selected checkout, interpreter, config, roots and launch context. For import
failures check the imported package path and working directory first; for access
failures distinguish missing files, authentication and model terms, and never substitute
another model silently. Linked skills live in the installed catalog or in the checkout's
`skills/` directory.
