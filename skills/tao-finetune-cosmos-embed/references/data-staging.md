# Data staging for Cosmos-Embed1

The `cosmos-embed1` CLI consumes local paths and Python globs, not raw `s3://.../*.mp4` URIs. For object-store-backed runs, first stage a subset or the full dataset to the execution host or container filesystem, then use local paths such as `/data/video/*.mp4` in the spec.

## MSR-VTT-style data

Recommended layout for staged MSR-VTT data:

```text
s3://bucket/path/cosmos-embed/msrvtt-subset/
├── msrvtt_test_1k.json
└── video/
    ├── video7020.mp4
    └── ...
```

After downloading or syncing that prefix into the mounted `data/` directory, point `dataset.*.mp4_urls` at `/data/video/*.mp4` and `dataset.*.metadata` at `/data/msrvtt_test_1k.json`, then use the Docker commands from `SKILL.md`.

## Chunk manifests (`vad_r1_chunks`)

Stage the manifest and the parent videos together and keep the manifest's relative prefix intact:

```text
s3://bucket/path/cosmos-embed/event-videos/
├── labels.json            # chunk manifest; video paths look like videos/<file>.mp4
└── videos/
    ├── <file>.mp4
    └── ...
```

The loader only rewrites paths through `dataset.*.path_prefix_mapping`, so map the manifest prefix to the staged folder, for example `{"videos/": "/data/event-videos/videos/"}`. Entries whose parent video is missing are skipped with a warning when `skip_missing_files` is true (the default); the training log reports how many videos were kept and dropped.

## Retrieval benchmark release root

`dataset_type: retrieval_benchmark` expects one directory holding `gt/`, `additional_gt/`, and `clips/`. Stage the whole root and pass it as `metadata`. The gallery is embedded on every validation pass, so keep it on fast local storage.
