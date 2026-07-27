# Datasets Reference — UC1 / UC2 / UC3

Use this reference only when the parent `SKILL.md` points here for the current task. If this file conflicts with current `SKILL.md`, `skill_info.yaml`, schemas, or platform/model skills, the current authoritative source wins.

## Contents

- UC1 — PCB
- UC2 — Metal Surface (Magnetic Tile)
- UC3 — Mobile Phone Screen
  - Step 1 — Download the Roboflow zip (manual, browser required)
  - Step 2 — Run the preparation script


How to obtain a ready-to-use `dataset_dir` for each supported use case.
The table below shows what is pre-packaged on Hugging Face and what must be prepared locally.

| UC | Subject | Anomaly types | HF provides | Local prep required |
|---|---|---|---|---|
| UC1 | PCB | bridge, missing, excess_solder | Full dataset | Run `prepare_dataset_uc1.py` |
| UC2 | Metal surface (Magnetic Tile) | MT_Blowhole, MT_Break, MT_Crack, MT_Fray, MT_Uneven | Nothing | Run `prepare_dataset_uc2.py` |
| UC3 | Mobile phone screen | oil, scratch, stain | masks + `defect_spec.jsonl` | Run `prepare_dataset_uc3.py` for images + masks |

Hugging Face authentication is required for UC1 and UC3. The caller needs
either `HF_TOKEN` exported (one-shot) or `hf auth login` run once (persists to
`~/.cache/huggingface`), with read access to the relevant `nvidia/Cosmos-AnomalyGen-*`
dataset repos. The `hf` CLI ships with the `cosmos-predict2` conda env that's
already active in the container; on the host install via
`pip install 'huggingface_hub>=1.0'`.

---

## UC1 — PCB

The complete UC1 dataset (anomaly images, masks, clean images, `defect_spec.jsonl`)
is shipped as a Hugging Face dataset repo.

**HF repo:** [`nvidia/Cosmos-AnomalyGen-PCB-Dataset`](https://huggingface.co/datasets/nvidia/Cosmos-AnomalyGen-PCB-Dataset)

**Run the preparation script:**

```bash
python3 -m scripts.utilities.prepare_dataset_uc1 <output_dir>
```

Optional flags:

- `--revision <rev>` — pin a different git revision / tag / commit (default: `main`)
- `--keep-download <dir>` — stage HF files in `<dir>` instead of a temp dir
- `--dry-run` — print the resolved HF target and exit

Pass `<output_dir>` as `dataset_dir` to the rest of the pipeline.

---

## UC2 — Metal Surface (Magnetic Tile)

The UC2 dataset is downloaded automatically from the public GitHub repository
[abin24/Magnetic-tile-defect-datasets](https://github.com/abin24/Magnetic-tile-defect-datasets).

**Run the preparation script:**

```bash
python3 -m scripts.utilities.prepare_dataset_uc2 <output_dir>
```

Optional — keep the raw zip for debugging:

```bash
python3 -m scripts.utilities.prepare_dataset_uc2 <output_dir> \
    --keep-download /tmp/magnetic_tile_raw
```

**Output layout:**

```
<output_dir>/
  metal_surface/
    anomaly_image/
      MT_Blowhole/   5 images
      MT_Break/      5 images
      MT_Crack/      5 images
      MT_Fray/       5 images
      MT_Uneven/     5 images
    mask/
      MT_Blowhole/   5 masks
      MT_Break/      5 masks
      MT_Crack/      5 masks
      MT_Fray/       5 masks
      MT_Uneven/     5 masks
    clean_image/     20 images
  defect_spec.jsonl
```

The script selects a curated subset (5 anomaly images + masks per type, 20 clean images)
matching the reference UC2 dataset. Pass `<output_dir>` as `dataset_dir`.

---

## UC3 — Mobile Phone Screen

The UC3 anomaly + clean images come from a Roboflow dataset. Masks and
`defect_spec.jsonl` are shipped as a Hugging Face dataset repo
([`nvidia/Cosmos-AnomalyGen-Glass-Masks`](https://huggingface.co/datasets/nvidia/Cosmos-AnomalyGen-Glass-Masks))
and can be fetched in the same script invocation.

### Step 1 — Download the Roboflow zip (manual, browser required)

Roboflow does not support unauthenticated programmatic download.

Follow the instructions in [`datasets/UC3_dataset_download_instructions.pdf`](UC3_dataset_download_instructions.pdf) to download the zip from
`https://universe.roboflow.com/vu-thi-thu-huyen/mobile-screen`.

### Step 2 — Run the preparation script

```bash
python3 -m scripts.utilities.prepare_dataset_uc3 <output_dir> \
    --zip <path/to/downloaded.zip> --masks-from-hf
```

Preview without writing files:

```bash
python3 -m scripts.utilities.prepare_dataset_uc3 <output_dir> \
    --zip <path/to/downloaded.zip> --masks-from-hf --dry-run
```

To pull just the masks/defect_spec (e.g., when the images already exist):

```bash
python3 -m scripts.utilities.prepare_dataset_uc3 <output_dir> --masks-from-hf
```

**Final output layout (after both flags):**

```
<output_dir>/
  Phone/
    anomaly_image/
      oil/       5 images  (from Roboflow)
      scratch/   5 images  (from Roboflow)
      stain/     5 images  (from Roboflow)
    clean_image/            (from Roboflow)
    mask/
      oil/                  (from HF)
      scratch/              (from HF)
      stain/                (from HF)
  defect_spec.jsonl         (from HF)
```

Pass `<output_dir>` as `dataset_dir` once the script completes.
