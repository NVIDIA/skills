# NeMo-Gym Benchmark Patterns Reference

Detailed patterns, schemas, and code examples for adding benchmarks to NeMo-Gym. Read this file when implementing a benchmark.

## Table of Contents

1. [Resources Server Pattern](#resources-server-pattern)
2. [YAML Config Pattern](#yaml-config-pattern)
3. [JSONL Data Schema](#jsonl-data-schema)
4. [Agent Patterns](#agent-patterns)
5. [Code Extraction Patterns](#code-extraction-patterns)
6. [Subprocess Execution with Ray](#subprocess-execution-with-ray)
7. [Test Patterns](#test-patterns)
8. [Data Conversion Script Pattern](#data-conversion-script-pattern)
9. [Dataset Registry Pattern](#dataset-registry-pattern)

---

## Resources Server Pattern

### Minimal verify-only server (e.g. `example_single_tool_call`)

```python
from nemo_gym.base_resources_server import (
    SimpleResourcesServer, BaseResourcesServerConfig,
    BaseVerifyRequest, BaseVerifyResponse,
)

class MyConfig(BaseResourcesServerConfig):
    pass

class MyServer(SimpleResourcesServer):
    config: MyConfig

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        model_output = body.response.output_text
        expected = (body.verifier_metadata or {}).get("expected_answer")
        reward = 1.0 if model_output.strip() == expected else 0.0
        return BaseVerifyResponse(**body.model_dump(), reward=reward)

if __name__ == "__main__":
    MyServer.run_webserver()
```

### Subprocess execution server (e.g. `code_gen`)

```python
from asyncio import Semaphore, get_running_loop
from time import time
from typing import Any, Dict, List, Optional
import ray
from nemo_gym.base_resources_server import (
    SimpleResourcesServer, BaseResourcesServerConfig,
    BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse,
)

class MyConfig(BaseResourcesServerConfig):
    num_processes: int = 8
    timeout_secs: int = 30
    debug: bool = False

class MyVerifyRequest(BaseRunRequest, BaseVerifyRequest):
    verifier_metadata: Optional[Dict[str, Any]] = None

class MyVerifyResponse(BaseVerifyResponse):
    extracted_code: Optional[str] = None
    # ... benchmark-specific result fields

class MyServer(SimpleResourcesServer):
    config: MyConfig

    def model_post_init(self, context):
        self._semaphore: Semaphore = Semaphore(value=self.config.num_processes)

    async def verify(self, body: MyVerifyRequest) -> MyVerifyResponse:
        model_out = body.response.output_text
        if not model_out or not model_out.strip():
            return MyVerifyResponse(**body.model_dump(), reward=0.0)

        code = extract_code(model_out)  # your extraction function
        if not code:
            return MyVerifyResponse(**body.model_dump(), reward=0.0)

        async with self._semaphore:
            future = run_tests_remote.remote(code, body.verifier_metadata)
            result = await future

        return MyVerifyResponse(
            **body.model_dump(),
            reward=1.0 if result["all_passed"] else 0.0,
            extracted_code=code,
        )
```

### Key rules

- `verify()` must be async
- Return `reward` as 0.0 or 1.0 (binary for RL)
- Handle empty/missing model output gracefully (return 0.0, don't crash)
- `verifier_metadata` is the opaque dict from JSONL — define whatever fields your benchmark needs
- Guard optional nested fields: `(body.verifier_metadata or {}).get("key", default)`
- Use `asyncio.Semaphore` to bound concurrent subprocess/external calls
- For Ray remote tasks: `result = await future` (Ray futures are directly awaitable). Never call `ray.get()` in async context.

---

## External Tool Auto-Install Pattern

When a benchmark requires an external tool (compiler, runtime, etc.), auto-install it so users don't need manual setup.

### setup module (`setup_<tool>.py`)

```python
import logging, os, shutil, subprocess, sys
from pathlib import Path

LOG = logging.getLogger(__name__)
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_PREFIX = _SCRIPT_DIR / ".toolname"
_INSTALL_SCRIPT = _SCRIPT_DIR / "scripts" / "install_tool.sh"

def ensure_tool() -> None:
    if shutil.which("tool"):
        LOG.info("tool found: %s", shutil.which("tool"))
        return
    LOG.warning("tool not found on PATH — attempting auto-install")
    if sys.platform == "darwin":
        subprocess.run(["brew", "install", "tool-package"], check=True)
    elif sys.platform == "linux":
        prefix = os.environ.get("TOOL_PREFIX", str(_DEFAULT_PREFIX))
        env = os.environ.copy()
        env["TOOL_PREFIX"] = prefix
        subprocess.run(["bash", str(_INSTALL_SCRIPT)], check=True, env=env)
        os.environ["PATH"] = str(Path(prefix) / "bin") + os.pathsep + os.environ.get("PATH", "")
        os.environ["LD_LIBRARY_PATH"] = str(Path(prefix) / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    else:
        raise NotImplementedError(f"Unsupported platform: {sys.platform}")
    # Verify
    if not shutil.which("tool"):
        raise RuntimeError("Install completed but tool still not on PATH")
```

### Server integration (`app.py`)

```python
def model_post_init(self, context):
    ensure_tool()  # auto-install before any requests
    self._semaphore = Semaphore(value=self.config.num_processes)
```

### Test integration (`tests/conftest.py`)

`pytest.mark.skipif` evaluates at **module import time** (during collection), before any fixtures run. To auto-install before skip checks, use `pytest_configure`:

```python
from setup_tool import ensure_tool

def pytest_configure(config):
    try:
        ensure_tool()  # runs before test collection
    except Exception:
        pass
    # Now skipif(shutil.which("tool") is None) will find the tool
```

### Build script (`scripts/install_tool.sh`)

- Versions as variables at the top
- Configurable prefix via env var (default: `../.toolname/`)
- Idempotent: skip steps if artifacts already exist
- Check for prerequisites (`gcc`, `make`, `wget`/`curl`)

### Gitignore

Add `.toolname/` to the server's `.gitignore`.

---

## YAML Config Pattern

A single YAML file in `configs/` defines both the resources server and its agent pairing(s):

```yaml
# Resources server instance
my_benchmark:
  resources_servers:
    my_benchmark:                    # must match subdirectory name
      entrypoint: app.py
      domain: coding                 # or: math, other
      # server-specific config fields:
      num_processes: 8
      timeout_secs: 30
      debug: false

# Simple agent pairing (single-turn)
my_benchmark_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_benchmark           # matches instance name above
      model_server:
        type: responses_api_models
        name: policy_model           # defined in env.yaml or policy config
      datasets:
      - name: my_dataset
        type: train
        jsonl_fpath: resources_servers/my_benchmark/data/my_dataset.jsonl
        gitlab_identifier:
          dataset_name: my_benchmark
          version: 0.0.1
          artifact_fpath: my_dataset.jsonl
        license: MIT                 # required for train/validation
        num_repeats: 1
      - name: example
        type: example
        jsonl_fpath: resources_servers/my_benchmark/data/example.jsonl

# Optional: custom agent pairing (multi-turn)
my_benchmark_eval_agent:
  responses_api_agents:
    my_eval_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_benchmark
      model_server:
        type: responses_api_models
        name: policy_model
      max_correction_turns: 3
      datasets:
      - name: my_dataset
        type: train
        jsonl_fpath: resources_servers/my_benchmark/data/my_dataset.jsonl
        gitlab_identifier:
          dataset_name: my_benchmark
          version: 0.0.1
          artifact_fpath: my_dataset.jsonl
        license: MIT
        num_repeats: 1
      - name: example
        type: example
        jsonl_fpath: resources_servers/my_benchmark/data/example.jsonl
```

### Key rules

- The `verified: false` flag is auto-added by pre-commit hook. Set to `true` after baselining.
- `license` is required for `train` and `validation` datasets.
- Valid license values: `Apache 2.0`, `MIT`, `CC-BY-4.0`, etc.
- `domain` should be one of: `coding`, `math`, `other`, or check `config_types.py` for current enum.
- Dataset `type` must be one of: `train`, `validation`, `example`.
- `gitlab_identifier` is required for `train`/`validation` datasets. `jsonl_fpath` is the local download path. Both fields coexist.
- `example` datasets don't have `gitlab_identifier` — they're committed to git directly.

---

## JSONL Data Schema

Each line is a JSON object with this structure:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "System prompt here"},
      {"role": "user", "content": "Problem description here"}
    ]
  },
  "verifier_metadata": {
    "test_cases": [...],
    "task_id": "unique_id",
    "category": "optional_category"
  }
}
```

### Required fields

- `responses_create_params.input`: list of message dicts with `role` and `content`. Follows OpenAI message format.

### Optional fields

- `responses_create_params.tools`: function tool definitions if the benchmark uses tool calls
- `responses_create_params.temperature`, `max_output_tokens`, etc.
- `verifier_metadata`: arbitrary dict passed through to `verify()`. Define whatever your benchmark needs.
- Any other top-level fields — they pass through to the resources server via `BaseRunRequest`.

### Data conversion

Convert from source format to Gym JSONL using a conversion script. See [Data Conversion Script Pattern](#data-conversion-script-pattern) for the script template and [Dataset Registry Pattern](#dataset-registry-pattern) for the upload workflow.

---

## Agent Patterns

### Simple agent (default — no custom code needed)

For single-turn benchmarks (model generates once, verify). Just reference `simple_agent` in YAML config.

### Multi-turn correction agent (e.g. `proof_refinement_agent`)

For benchmarks where the model gets error feedback and retries:

```python
from nemo_gym.base_responses_api_agent import (
    BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import raise_for_status

class MyAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef
    max_correction_turns: int = 3

class MyAgent(SimpleResponsesAPIAgent):
    config: MyAgentConfig

    async def responses(self, request, response, body=Body()):
        # Forward to model server, return NeMoGymResponse
        model_response = await self.server_client.post(
            server_name=self.config.model_server.name,
            url_path="/v1/responses",
            json=body,
            cookies=request.cookies,
        )
        await raise_for_status(model_response)
        return NeMoGymResponse.model_validate(await model_response.json())

    async def run(self, request, body):
        cookies = request.cookies

        # 1. Seed session
        seed = await self.server_client.post(
            self.config.resources_server.name, "/seed_session",
            json=body.model_dump(), cookies=cookies,
        )
        cookies = seed.cookies

        current_input = body.responses_create_params
        for turn in range(self.config.max_correction_turns + 1):
            # 2. Generate
            gen = await self.server_client.post(
                self.config.name, "/v1/responses",
                json=current_input, cookies=cookies,
            )
            cookies = gen.cookies
            model_json = await gen.json()

            # 3. Verify
            verify_data = body.model_dump()
            verify_data["response"] = model_json
            verify = await self.server_client.post(
                self.config.resources_server.name, "/verify",
                json=verify_data, cookies=cookies,
            )
            cookies = verify.cookies
            result = await verify.json()

            if result.get("reward", 0.0) == 1.0:
                break
            if turn >= self.config.max_correction_turns:
                break

            # 4. Build correction prompt from errors
            current_input = {"input": [{"role": "user", "content": build_correction(result)}]}

        return result
```

### Key rules

- Propagate cookies through every server call: `cookies=request.cookies`, then update with `response.cookies`. This is critical for stateful environments that track per-task sessions.
- Call `raise_for_status()` after every inter-server call
- The agent calls itself (`self.config.name`) for `/v1/responses` to keep middleware chain intact
- Use `ConfigDict(extra="allow")` on request/response models for flexible field forwarding
- **Token ID propagation**: model responses include `prompt_token_ids`, `generation_token_ids`, and `generation_log_probs`. Propagate these from each model response into the next turn's input — required for RL training. See `swe_agents/app.py` for the implementation pattern.
- **Monotonic trajectories**: NeMo RL requires the token sequence across multi-turn rollouts to only grow. Never summarize, truncate, or modify prior turns between steps.
- **Thinking model compatibility**: thinking models emit `<think>`/`<thinking>` blocks. Strip these before parsing tool calls from model output, or parsing will break. See `harbor_agent` for the handling pattern.
- **Concurrency control for external services**: if the agent calls external services (Docker containers, APIs), use `asyncio.Semaphore` to throttle concurrent calls — external services may not handle thousands of simultaneous requests.

---

## Code Extraction Patterns

For benchmarks that need to extract code from model output:

1. Strip `<think>`/`<thinking>` blocks (reasoning traces from thinking models)
2. Extract from markdown code fences (` ```lang ``` `)
3. Fall back to language-specific markers
4. Validate extracted code has required structure

Handle these edge cases:
- Orphaned closing tags (`</think>` without matching `<think>`)
- Unclosed code fences
- Multiple code blocks (pick longest)
- No code fences at all (raw code in response)

---

## Subprocess Execution with Ray

For benchmarks that compile/run code:

```python
import ray
import subprocess
import tempfile
from pathlib import Path

@ray.remote
def run_tests_remote(code, test_cases, timeout=30):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write source, compile, run tests
        src = Path(tmpdir) / "program.ext"
        src.write_text(code)

        try:
            result = subprocess.run(
                ["compiler", str(src)],
                capture_output=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"all_passed": False, "error": "timeout"}
        except FileNotFoundError:
            return {"all_passed": False, "error": "compiler not found"}

        # ... run test cases, compare output
        return {"all_passed": all_passed, ...}
```

### Key rules

- Decode subprocess output with `errors="replace"` to handle non-UTF8
- Implement fail-fast after N consecutive timeouts
- Handle `FileNotFoundError` for missing compilers/tools
- Use `tempfile.TemporaryDirectory` for isolation
- Executables must run on Linux

---

## Test Patterns

### Resources server tests

```python
import shutil
import pytest
from unittest.mock import MagicMock
from nemo_gym.server_utils import ServerClient
from resources_servers.my_benchmark.app import MyServer, MyConfig

SKIP_REASON = "tool-name not installed"

@pytest.mark.skipif(
    shutil.which("tool-name") is None, reason=SKIP_REASON
)
class TestMyServer:
    def setup_method(self):
        self.config = MyConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        self.server = MyServer(config=self.config, server_client=MagicMock(spec=ServerClient))

    # Test cases: verify_pass, verify_fail_compile, verify_fail_wrong_output,
    #             verify_no_code, verify_timeout, code_extraction
```

### Agent tests

Mock `server_client.post()` to simulate server responses without running actual servers.

---

## Data Conversion Script Pattern

Conversion scripts and prompt files belong in the **source repo** (e.g. your dataset repository), not in NeMo-Gym. Only the converted JSONL files are uploaded to the GitLab dataset registry.

**Exception**: When there is no external source repo, keep the conversion script in the resources server directory.

Reference script template (for use in the source repo):

```python
import argparse
import json
from pathlib import Path

def convert_problem(problem: dict, system_prompt: str) -> dict:
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem["prompt"]},
            ]
        },
        "verifier_metadata": {
            "test_cases": problem["tests"],
            "task_id": problem["id"],
        },
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--system-prompt", required=True, help="Path to system prompt text file")
    parser.add_argument("--example-output", help="Path to example.jsonl (first 5 entries)")
    args = parser.parse_args()

    system_prompt = Path(args.system_prompt).read_text().strip()

    with open(args.input) as f:
        data = json.load(f)

    with open(args.output, "w") as out:
        for problem in data:
            record = convert_problem(problem, system_prompt)
            out.write(json.dumps(record) + "\n")

    if args.example_output:
        with open(args.input) as f:
            data = json.load(f)
        with open(args.example_output, "w") as out:
            for problem in data[:5]:
                record = convert_problem(problem, system_prompt)
                out.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    main()
```

Externalize system prompts to text files, pass via `--system-prompt` argument. Multiple prompt tiers enable ablation studies.

---

## Dataset Registry Pattern

### Dataset types and where they live

| Type | Location | Committed to git? | `gitlab_identifier`? |
|------|----------|-------------------|---------------------|
| `example` | `data/example.jsonl` | Yes | No |
| `train` | GitLab dataset registry | No | Yes |
| `validation` | GitLab dataset registry | No | Yes |

### `data/.gitignore` default patterns

Generated by `ng_init_resources_server`:
```
*train.jsonl
*validation.jsonl
*train_prepare.jsonl
*validation_prepare.jsonl
*example_prepare.jsonl
```

If your filename doesn't match (e.g. `my_eval.jsonl`), add a custom pattern (e.g. `*eval.jsonl`).

### Upload workflow

1. **Generate** the JSONL file using your conversion script (in the source repo)
2. **Upload** to GitLab dataset registry:
   ```bash
   ng_upload_dataset_to_gitlab \
       +dataset_name=my_benchmark \
       +version=0.0.1 \
       +input_jsonl_fpath=resources_servers/my_benchmark/data/my_dataset.jsonl
   ```
3. **Add `gitlab_identifier`** to the dataset entry in YAML config:
   ```yaml
   - name: my_dataset
     type: train
     jsonl_fpath: resources_servers/my_benchmark/data/my_dataset.jsonl
     gitlab_identifier:
       dataset_name: my_benchmark
       version: 0.0.1
       artifact_fpath: my_dataset.jsonl
     license: MIT
   ```
4. **Ensure `.gitignore`** covers the filename (add custom pattern if needed)
5. **Remove from git** if previously tracked: `git rm --cached <file>`

### MLflow credentials

Upload/download requires MLflow credentials in `env.yaml`:
```yaml
mlflow_tracking_uri: <your-gitlab-mlflow-tracking-uri>
mlflow_tracking_token: <your-gitlab-api-token>
```

The tracking URI format is `https://<gitlab-host>/api/v4/projects/<PROJECT_ID>/ml/mlflow`.

### Verification with `ng_prepare_data`

Validate example data (for PR submission):
```bash
ng_prepare_data "+config_paths=[resources_servers/my_benchmark/configs/my_benchmark.yaml]" \
    +output_dirpath=/tmp/prepare +mode=example_validation
```

Download and prepare train/validation from GitLab:
```bash
ng_prepare_data "+config_paths=[resources_servers/my_benchmark/configs/my_benchmark.yaml]" \
    +output_dirpath=data/my_benchmark +mode=train_preparation +should_download=true +data_source=gitlab
```

---

## External Benchmark Integration

When wrapping a 3rd-party benchmark library (e.g. SWE-bench, BigCodeBench), integrate at the agent server level:

```python
class ExternalBenchmarkAgent(SimpleResponsesAPIAgent):
    config: ExternalBenchmarkAgentConfig

    async def responses(self, request, response, body=Body()):
        raise NotImplementedError("Use /run endpoint directly")

    async def run(self, request, body):
        # 1. Pre-process: Gym schema → library input
        library_input = preprocess(body)

        # 2. Call external library
        library_result = await run_external(library_input)

        # 3. Post-process: library result → BaseVerifyResponse
        return BaseVerifyResponse(
            **body.model_dump(),
            reward=1.0 if library_result.passed else 0.0,
            response=library_result.response,
        )
```

Add the dependency in `requirements.txt`. If needs are more complex than pip packages, use `setup.py` or `pyproject.toml`.

Reproduction requirement: run the original repo first, reproduce published numbers, then integrate into Gym and reproduce again. This decouples Gym integration bugs from benchmark bugs.

### Replacing httpx with aiohttp in external libraries

Many external libraries use `httpx.AsyncClient` internally. NeMo Gym requires all async HTTP to go through its global aiohttp client (`nemo_gym.server_utils.request()`) because httpx/httpcore has O(n^2) connection pooling that causes hangs at high concurrency. See `docs/infrastructure/engineering-notes/aiohttp-vs-httpx.md`.

When wrapping such a library, create an adapter class that mimics the library's expected HTTP interface but routes calls through aiohttp. The `tavily_search` resources server demonstrates this pattern:

```python
from nemo_gym.server_utils import request

class AIOHTTPAdapter(BaseModel):
    headers: Dict[str, str]
    base_url: str

    async def post(self, endpoint: str, content: str, **kwargs) -> AdapterResponse:
        response = await request(
            method="POST",
            url=f"{self.base_url}{endpoint}",
            headers=self.headers,
            data=content,
        )
        return AdapterResponse(status_code=response.status, data=await response.json())

    @classmethod
    def from_httpx_client(cls, client: AsyncClient) -> "AIOHTTPAdapter":
        return cls(headers=dict(client.headers), base_url=str(client.base_url))
```

Then in `model_post_init()`, replace the library's internal client:
```python
def model_post_init(self, context):
    self._external_client = ExternalLibraryClient(api_key=key)
    self._external_client._http_client = AIOHTTPAdapter.from_httpx_client(
        self._external_client._http_client
    )
```

---

## Reward Profiling Best Practices

### Model selection

Recommended model suite (as of Feb 2026):
- Your policy model of interest
- GPT-5 Nano, GPT-5 (closed-source baseline)
- Qwen 3 30B A3B Instruct 2507 (open-source instruct)
- Qwen 3 30B A3B Thinking 2507 (open-source thinking)
- If 30B models are too weak: Qwen 3 235B A22B variants, Kimi K2 Instruct, or GLM-4.7

### Validation checks

- Open-source models should achieve 30%+ on your benchmark. If not, investigate bugs.
- Closed-source models should score at or above open-source models. The reverse is rare and usually indicates a bug.
- Run both instruct and thinking models — the benchmark should be model-agnostic.
- Inspect actual failure cases in the rollout JSONL, not just aggregate numbers.

### Variance control

Run the highest-scoring open-source model multiple times. Increase `num_repeats` until variance < 1%.

### Training environments

If adding a training environment (not just a benchmark), additionally run training with NeMo RL:
- GRPO algorithm, 64 prompts per step, 16 rollouts per prompt (adjustable)
- Train with both instruct and thinking models
- Include W&B links and train/validation curve screenshots in the PR

---

## Token ID Propagation (Multi-Turn Training)

During training, model responses include `prompt_token_ids`, `generation_token_ids`, and `generation_log_probs` on response messages. When constructing multi-turn input for the next model call, propagate these fields from the previous model response. This is required for RL training to work correctly.
