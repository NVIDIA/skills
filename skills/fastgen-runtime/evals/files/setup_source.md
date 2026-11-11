# FastGen source fixture

From `fastgen/utils/scripts.py::setup` at FastGen revision
`e0a92619fc56daf4dfa081273c7729c045bf1dcc`. Excerpts are in execution order;
intervening logging and checks are omitted. Read this source; do not execute it.

## fastgen/utils/scripts.py:154

```text
    # Check for dryrun
    if getattr(args, "dryrun", False):
        logger.info("Dryrun")
        logger.info(serialize_config(config, return_type="yaml"))
        sys.exit(0)
```

## fastgen/utils/scripts.py:162

```text
    distributed = ddp.init(required=fsdp_enabled or (ddp_enabled and not evaluation))
```

## fastgen/utils/scripts.py:183

```text
    # Global batch size
    if config.trainer.batch_size_global is not None:
        batch_size = config.dataloader_train.batch_size * world_size()
        accum_rounds = max(config.trainer.batch_size_global // batch_size, 1)
        new_batch_size_global = accum_rounds * batch_size
```
