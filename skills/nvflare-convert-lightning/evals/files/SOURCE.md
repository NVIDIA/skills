# Fixture Source Notes

The `hello-lightning` fixtures are minimized, unconverted PyTorch Lightning
training code modeled on the NVFLARE repository example:

- Source example: `examples/hello-world/hello-lightning`

The fixture intentionally omits real datasets, data download, FLARE integration,
and full job execution details so trigger and behavior evals stay deterministic.
`train.py` and `model.py` represent plain Lightning code before any FLARE
conversion; the agent under evaluation is expected to add the
`flare.patch(trainer)` Client API integration and a `job.py`.

The `gpu-device-lightning` fixture is synthetic, derived from
`hello-lightning` with an explicit `torch.cuda.is_available()` choice between
Lightning's `gpu` and `cpu` accelerators. It makes device-intent preservation
applicable without requiring a GPU on the evaluation host.

The `vocab-lightning` fixture adds a `LitTextCNN` model whose `__init__` has a
required, data-derived argument (`vocab_size`, no default). The conversion must
pin one shared vocabulary size for the server recipe model config and every
client model construction path. Passing a live `LightningModule` instance with
required args can serialize without those args and fail server-side
reconstruction in the model persistor.

The `external-data-lightning` fixtures are synthetic, derived from the
`hello-lightning` fixture but loading train/val CSVs from an external data
directory (`--data-dir`, default `/data/nvflare/lightning-tabular`) instead of
building synthetic in-memory tensors. The path is intentionally external to the
repository and run workspace so configurable data-path behavior is asserted only
when the source provides an external dataset location.

The `hello-lightning` fixture's `LitNet` includes `validation_step` with
`self.log("val_loss", ...)` and the training entry point builds a validation
dataloader, so evaluation-focused evals can assert Lightning-native evaluation
(`trainer.validate` before `trainer.fit`) without a separate fixture.

The `routing-dual` fixture contains independent Lightning and Hugging Face
Trainer entrypoints in one project. It verifies that project-wide dual training
ownership routes to `nvflare-orient` instead of either framework converter.
