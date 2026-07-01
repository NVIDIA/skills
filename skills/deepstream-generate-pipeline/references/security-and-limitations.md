# Security, Limitations & Notes

Reference details for the DeepStream Pipeline Builder skill. These do not affect the
interactive workflow — they document the security posture, known limitations, and
operational notes of the pipeline generator.

## Security

- **No shell execution of user input:** All subprocess calls use list-form arguments (`subprocess.run([...])`, never `shell=True`), preventing shell injection regardless of pipeline string content
- **Element name validation:** Element names extracted from the pipeline are validated against `[a-zA-Z0-9_-]` before being passed to `gst-inspect-1.0`; names that fail validation are skipped rather than passed to subprocess
- **Safe tokenization:** The live parse check uses `shlex.split()` rather than naive string splitting, so quoted tokens are handled correctly and unexpected argument injection is avoided
- **Output length cap:** GStreamer stderr lines are capped at 300 characters in warnings to prevent info leakage from verbose runtime output
- **Pipeline size limit:** Inputs longer than 16 384 characters are rejected before any processing
- **No credential handling:** This skill does not process, store, or transmit credentials, tokens, or secrets
- **Data classification:** Public — not intended for processing sensitive or confidential media files

## Limitations

- Requires DeepStream SDK installed locally for element validation (`gst-inspect-1.0`) and live parse checks (`gst-launch-1.0`)
- Pipeline dataset covers common DeepStream patterns — exotic or fully custom element chains may need manual assembly using the assembly rules in `references/assembly-rules.md`
- Live parse dry-run is skipped for multi-stream pipelines (those with named pad refs like `m.sink_0`) because `fakesrc` cannot negotiate caps through `nvstreammux` named pads
- Platform is Linux only — macOS and Windows are not supported
- The BM25 retriever uses no embeddings or semantic model; very unusual queries may get `confidence: low`, in which case the assembly rules in this skill take precedence over retrieved examples

## Notes

- The script is fully standalone — zero external dependencies, pure Python stdlib (BM25 scoring with structural metadata boosting and domain synonym expansion)
- The pipeline dataset (`data.csv`) contains 270+ verified DeepStream pipelines covering decode, encode, inference, tracking, format conversion, and more
- Default sample paths use `/opt/nvidia/deepstream/deepstream/samples/` — remind users to update paths for their setup
- For inference pipelines, users must provide their own `config_infer_*.txt` config files — the defaults point to DeepStream sample configs
- The skill assembles pipelines from proven patterns — it does NOT invent arbitrary element chains
- The retriever outputs a `confidence` field — when confidence is low, the LLM should rely more on the assembly rules than on the retrieved examples
