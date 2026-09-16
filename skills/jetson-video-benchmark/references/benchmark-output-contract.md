# Benchmark evidence

Keep the result small, readable, and independently checkable as a concise
evidence handoff.

## Minimal retained result

Before reporting a live benchmark complete, write one
`benchmark-results.json` in the run workspace and retain the raw output of each
launch in its own log file. The JSON needs only:

- status, target/release, selected surface, GPU, and clocks when relevant;
- input canonical path, bytes, SHA-256, format, geometry, FPS, and source frame
  count;
- recipe canonical path, bytes, SHA-256, and compared knob, for encode;
- authenticated launcher/sample path, bytes, SHA-256, exact argv, and timeout;
- warmup outcome and, for every measured repetition, reported frame count,
  elapsed time, FPS, MP/s when supported, plus raw-log paths and hashes;
- recomputed repetition count, mean, minimum, and maximum; and
- branch status, limitations, and failed attempts.

Keep package listings, setup details, SDK sources, build trees, and media
outside this file. Keep native and PyNvVideoCodec results separate.

## Acceptance checklist

Apply every relevant check before accepting the result:

1. Reopen and rehash the input, recipe/config, and launcher immediately before
   the series and after it. Their paths must be canonical regular files, not
   symlinks, and their recorded byte counts and SHA-256 values must still
   match.
2. For raw encode input, match width, height, frame count, FPS, and pixel format
   to the recipe. Do not compare the raw input's codec with the recipe codec:
   `raw` describes the source while `h264`, `hevc`, or `av1` describes encoder
   output. Never relabel raw input to satisfy an output-codec field.
3. Bind the launched native preset or Python `-json` config to that recipe. In
   a comparison, recipes and launch arguments differ only in the requested
   knob; P4/P5 therefore differs only in preset.
4. Use exactly the same checked argv for one excluded warmup and at least three
   sequential measured processes per variant. Apply the common 1,000-frame cap
   when a PyNv encode surface participates and report the effective aggregate
   frame count.
5. Each process exits zero without timeout or a failure marker, and its official
   sample marker reports the expected positive frame count and FPS. Wall time
   is not a substitute for sample-reported FPS.
6. Derive every retained metric from the raw logs. Require positive finite FPS;
   compute `MP/s = FPS * width * height / 1_000_000` only when the sample binds
   the dimensions; recompute mean, minimum, and maximum from the measured runs.

If any check fails, name the check and offending values, reject that branch,
and retain independently completed peers as `partial`. A result whose metrics
cannot be recomputed from the retained raw logs is not complete.

Documentation estimates are not live benchmark evidence. They set
`measurement_performed: false` and follow
[documented-performance-estimates.md](documented-performance-estimates.md).

A throughput result measures only the evidenced codec workload. It does not
prove zero copy, shared-buffer compatibility, absence of copies, or a usable
cross-stage synchronization primitive unless those properties were separately
traced and authenticated.

## Interpretation

- Never pool native and Python repetitions or rank an unspecified `auto`
  request.
- Omit MP/s for the released Python decode sample because it does not report
  dimensions.
- A preset comparison reports throughput only; it does not prove visual quality
  or compression ordering.
- A worker sweep reports only the maximum passing tested codec-stage bound, not
  a camera or end-to-end capacity.
