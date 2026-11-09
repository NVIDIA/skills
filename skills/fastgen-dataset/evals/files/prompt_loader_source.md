# FastGen source fixture

Source revision: `e0a92619fc56daf4dfa081273c7729c045bf1dcc`. Excerpts below are input evidence, not an installed checkout.

## fastgen/configs/data.py:14

```python
from fastgen.datasets.wds_dataloaders import (
    WDSLoader,
    ImageWDSLoader,
    VideoWDSLoader,
    AudioVideoWDSLoader,
    CameraVideoWDSLoader,
)
```

## fastgen/configs/data.py:22

```python
from fastgen.configs.instantiate import LazyCall as L
```

## fastgen/configs/data.py:59

```text
PromptLoaderConfig = L(WDSLoader)(
    datatags=["WDS:/path/to/prompts"],
    batch_size=1,
    key_map={"condition_raw": "txt"},
    presets_map={"neg_condition_raw": "empty_string"},
    num_workers=2,
    sampler_start_idx=None,
)
```

## fastgen/datasets/wds_dataloaders.py:335

```text

        Returns:
            Dict with 'fname', 'shard', and all mapped output keys.
        """
        output = {
            **init_output_dict(item),
            **self._constants,
        }

        # Extract keys from item, falling back to constants if present
        for output_key, item_key in self._key_map.items():
            if item_key in item:
                output[output_key] = item[item_key]
            else:
                # ``filter_items`` already dropped items that lack both the key and a constant fallback.
                assert (
                    output_key in self._constants
                ), f"Missing required key '{item_key}' for output '{output_key}', without a default preset value."
                logger.debug(f"Missing key '{item_key}' for output '{output_key}', using default preset value.")
        return output

    def filter_items(
        self,
        item: dict,
    ) -> bool:
        """Filter function for WebDataset items before decoding.

        Checks that all required keys from key_map are present (unless a constant
        fallback exists) and that present keys contain raw byte data.

        Args:
            item: Raw WebDataset item dict (pre-decoding, values are bytes).

        Returns:
            True if the item has the data to be processed.
        """
        for output_key, item_key in self._key_map.items():
            if item_key not in item:
                # Missing key is only OK if we have a constant fallback
                if output_key not in self._constants:
                    return False
            elif not isinstance(item[item_key], (bytes, bytearray)):
                # Present keys must be raw bytes (not yet decoded)
                return False
        return True

    def _prefilter(self, dataset: wds.WebDataset) -> wds.WebDataset:
        # presets_filter excludes samples on purpose, so it must not count toward failures; deterministic loading
        # rejects it (see validate_wds_config).
        for filter_fn in self._filter_fn:
            dataset = dataset.select(filter_fn)
        return dataset

```
