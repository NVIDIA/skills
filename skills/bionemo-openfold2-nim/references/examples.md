# OpenFold2 Examples

## Hosted Sequence-Only Smoke Test

```python
import os
import json
from pathlib import Path
import requests

seq = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPT"
url = "https://health.api.nvidia.com/v1/biology/openfold/openfold2/predict-structure-from-msa-and-template"
headers = {
    "Authorization": f"Bearer {os.getenv('NGC_API_KEY')}",
    "Content-Type": "application/json",
}
payload = {
    "sequence": seq,
    "input_id": "kras_fragment",
    "selected_models": [1],
    "relax_prediction": False,
}
r = requests.post(url, headers=headers, json=payload, timeout=300)
r.raise_for_status()
result = r.json()
Path("openfold2_response.json").write_text(json.dumps(result, indent=2))
print(result.keys())
```

## Hosted With A3M

```python
seq = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPT"
payload = {
    "sequence": seq,
    "input_id": "kras_with_msa",
    "selected_models": [1, 2],
    "alignments": {
        "uniref90": {
            "a3m": {
                "alignment": f">query\n{seq}\n>homolog1\nMTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPT",
                "format": "a3m",
            }
        }
    },
}
```

## Local Request

```python
import requests

url = "http://localhost:8000/biology/openfold/openfold2/predict-structure-from-msa-and-template"
payload = {
    "sequence": "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPT",
    "input_id": "local_kras_fragment",
    "selected_models": [1],
}
r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=300)
r.raise_for_status()
print(r.json().keys())
```

## Explicit mmCIF Template Skeleton

Use a real template mmCIF for production. This pattern shows the field shape:

```python
payload["use_templates"] = True
payload["explicit_templates"] = [{
    "structure": Path("template.cif").read_text(),
    "format": "mmcif",
    "name": "template_1",
    "source": "user_provided",
}]
```
