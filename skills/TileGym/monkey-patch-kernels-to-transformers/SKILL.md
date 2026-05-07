---
name: monkey-patch-kernels-to-transformers
description: Integrate TileGym kernels to HuggingFace `transformers` models by replacing that library's submodule(s) and certain class(es)' implementation, and patching certain class(es)'s init/forward/load weight prior to instantiate models. Used when user requires integrate TileGym kernels to `transformers` library model by monkey patch approach.
version: 2026.04.16-beta
environment:
  IDE:
  - Claude Code
  - Cursor (Agent mode)
  model:
  - Opus 4.6
  - GPT-5.3-CodeX
license: CC-BY-4.0 AND Apache-2.0
metadata:
  author: "TileGym Team <TileGym@nvidia.com>"
  tags:
    - tilegym
    - transformers
    - integration
    - kernel
    - monkey-patch
---
# Integrate TileGym kernels to Transformers
The main purpose of TileGym project is to provide performant kernels for LLM training and inference. We will integrate proper kernels available in TileGym project to LLM models provided by Hugging Face `transformers` library to validate end-to-end functional correctness and performance improvements. Instead of modifying `transformers` source code, we will take a non-intrusive monkey-patch approach: We will replace certain modules/classes/methods in `transformers` library that implement the Transformer model we would like to integrate, such that at model instantiation, that model's core components will be replaced by TileGym implementations. At runtime the model will actually invoke TileGym kernels under the hood.

## Instructions
The integration process follows a "research kernel requirement and supply -> propose kernel integration candidates -> implement kernel integrations and verify -> aggregate valid integrations" workflow. Refer to the diagram below to understand the overall process, then check the numbered text below for details. If you find it difficult to interpret embedding Mermaid script, check the rendered PNG image which represents the exactly identical workflow diagram:
<details>

![Kernel integration workflow](./references/workflow-diagram.png)
</details>

```mermaid
flowchart TD
  %% Nodes are labeled ONLY by step number; read the numbered text below for details.
  %% Styling encodes who executes the step (orchestrator vs subagent).

  classDef orch fill:#E3F2FD,stroke:#1E88E5,stroke-width:2px,color:#0D47A1;
  classDef sub fill:#FFF3E0,stroke:#FB8C00,stroke-width:2px,color:#E65100;
  classDef decision fill:#E8F5E9,stroke:#43A047,stroke-width:2px,color:#1B5E20;
  classDef terminal fill:#F3E5F5,stroke:#8E24AA,stroke-width:2px,color:#4A148C;

  S([Start]):::terminal
  E([End]):::terminal

  step1[Step 1]:::orch
  join1(( )):::orch

  step2{Step 2}:::decision
  step2_1[Step 2.1]:::orch
  step2_2[Step 2.2]:::orch

  step3[Step 3]:::orch
  step3_1[Step 3.1]:::orch
  step3_2[Step 3.2]:::orch
  step3_3{Step 3.3}:::decision
  step3_4[Step 3.4]:::orch
  step3_5[Step 3.5]:::orch

  %% Explore subagents for Step 1 (delegated research; two distinct agents)
  subgraph subagent_explore_1
    %% Explore subagent A: Step 1.1
    step1_1[Step 1.1]:::sub
  end

  subgraph subagent_explore_2
    %% Explore subagent B: Step 1.2
    step1_2[Step 1.2]:::sub
  end

  %% Explore subagents (research phase)
  S --> step1
  step1 -->|parallel| step1_1
  step1 -->|parallel| step1_2
  step1_1 --> join1
  step1_2 --> join1
  join1 --> step2

  %% Plan phase branching
  step2 -->|already patched| E
  step2 -->|needs patching| step2_1 --> step2_2 --> step3

  %% Execute-and-verify phase (orchestrator)
  step3 --> step3_1 --> step3_2

  %% Code subagent per integration-plan item (sub-workflow for Step 3.2)
  subgraph subagent_code
    %% Code subagent loop (runs once per integration-plan item)
    step3_2_1[Step 3.2.1]:::sub
    step3_2_2[Step 3.2.2]:::sub
    step3_2_3[Step 3.2.3]:::sub
    step3_2_4{Step 3.2.4}:::decision
    step3_2_5{Step 3.2.5}:::decision
    step3_2_6[Step 3.2.6]:::sub
    step3_2_7[Step 3.2.7]:::sub

    step3_2_1 --> step3_2_2 --> step3_2_3 --> step3_2_4
    step3_2_4 -->|no candidates| step3_2_7
    step3_2_4 -->|candidate selected| step3_2_5
    step3_2_5 -->|mismatch: invalidate candidate| step3_2_4
    step3_2_5 -->|match| step3_2_6 --> step3_2_7
  end

  %% Orchestrator iterates items; accepts/rejects subagent output
  step3_2 --> step3_2_1
  step3_2_7 -->|next plan item| step3_2
  step3_2 -->|all items attempted| step3_3

  %% Aggregate + finalize, or exit if nothing viable
  step3_3 -->|none verified| E
  step3_3 -->|some verified| step3_4 --> step3_5 --> E
```

- Mapping note: `Step 1.1/1.2` correspond to the two explore-subagent bullets under Step 1; `Step 2.1/2.2` correspond to the two plan sub-steps under Step 2; `Step 3.2.1-3.2.7` correspond to the code-subagent sub-steps under Step 3.2.

### Detailed Steps
1. Research phase: Study the target Transformer model and available kernel and monkey-patch implementation in TileGym. Launch 2 parallel explore subagents:
   * Search the model ID on HuggingFace to know what architectures does it use. Then search GitHub code to get implementation of that architecture. Go through details to understand computations performed on every components. Summarize a comprehensive requirement list with all necessary details included. *Focus on details*. Some model might use variants of standard Attention/MoE/normalization, and/or use distinct data types at different part of computations;
   * Go through @src/tilegym/ to inventory available kernel implementations, OP interfaces, and Transformer model monkey-patches. Pay attention to the `@dispatch("<OP name>")` and `@register("<OP name>")` mappings, and `apply_tilegym_kernel_to_<transformer_module>` patch patterns. Summarize a manifest that list all available monkey-patch functions, OP interfaces, kernel implementations with sufficient details to distinguish variants of operations. *Refer to but don't rely on docstring/comments; focus on details that distinct similar kernels*. If unsure about `cuda.tile` kernel semantic, check https://docs.nvidia.com/cuda/cutile-python/operations.html.
2. Plan phase: Check if the target model architecture is already patched. If so, inform the user and exit; Otherwise, propose an integration plan following these sub-steps:
   1. Check the requirement list and manifest to determine which set of computations could be patched by TileGym implementations. Be optimistic since subsequent steps/subagents will drop unsuitable proposals;
   2. For each of the computation selected at previous sub-step, propose matching TileGym OP interfaces or/and concrete kernel implementations. You may propose multiple candidates if uncertain, but do keep candidate pool small using your best judgement.
3. Execute-and-verify phase: Prepare develop environment, launch subagents to implement monkey-patch for each of the items in integration plan once-a-time, verify it on develop environment, and accept/reject that monkey-patch. Specific sub-steps:
   1. The orchestrator agent (i.e., you) prepares a GPU develop environment by following [environment-setup.md](./references/environment-setup.md). This environment will be used by subsequent subagents.
   2. For each of the unverified integration plan item (i.e., a mapping of Transformer model compute <-> one or more TileGym implementation candidates), launch a parallel code subagent with VERY STRONG plan-following and debugging ability. Tell this subagent the allocated node name at sub-step 3.1. Workflow of this code subagent is:
      1. Study src/tilegym/transformers/monkey_patch.py and modeling/transformers/infer.py to understand how to monkey-patch a transformer compute with TileGym implementation. Study [docker-gpu-guide.md](./references/docker-gpu-guide.md). If any subsequent step requires executing a script, always SSH to the node name given by the orchestrator agent, build or rebuild image by modeling/transformers/Dockerfile if needed, and execute that script in the NVIDIA Docker environment;
      2. Locate the integration point at `transformers` library. E.g., It could be a `nn.Module` subclass that corresponds to a layer in the transformer model, or an utility function that applies certain modification to transformer models' intermediate variables/tensors. Use GitHub MCP search_code for unseen imports;
      3. Collect inputs and outputs around integration point to serve as subsequent verifications' references. You can create a simple debug Python script that calls `transformers` library's `.generate()` API to prompt the Transformer model to output "The capital of France is", and add code before and after the integration point to save intermediate PyTorch tensors and other necessary variables to disk as future references. *Critical: unoptimized `.generate()` is slow, collect as less data as possible*;
      4. Select the next unverified TileGym implementation candidate. If no unverified candidate available, exit current subagent and let the orchestrator agent know that the current Transformer compute is unsuitable for TileGym to patch; Otherwise, implement a monkey-patch function following the convention studied at sub-step 3.2.1. The patch function of current compute goes to src/tilegym/transformers/<submodule_name>/monkey_patch_<compute_name>.py. If additional modifications are need for the current transformer model (similar to the scenario of src/tilegym/transformers/deepseek2/modeling_deepseek.py), check existence (create by other subagents) or create a self-contained Python submodule src/tilegym/transformers/<submodule_name>/modeling_<submodule_name>.py and place modifications there;
      5. Verify the monkey-patch implementation at sub-step 3.2.4 by creating a Python script that instantiate a submodule that contains integration point, apply the monkey-patch, feed input data collected at sub-step 3.2.3, and collect output data. The output data should match the reference output collected at sub-step 3.2.3 within a reasonable error tolerance. Try your best to fix errors caused by integration and to resolve mismatch. If can't fix, mark current TileGym implementation candidate as invalid and go back to sub-step 3.2.4; Otherwise continue to next sub-step;
      6. Consolidate the debug and test code you implemented to src/tilegym/transformers/<submodule_name>/test_monkey_patch_<compute_name>.py and organize it in pytest style and remove all other files/scripts/documents/binary data files you created during debugging. Ensure only left one test case that checks input-output around the integrating point match with those from origin implementation and ensure the test case pass. At this point, src/tilegym/transformers/<submodule_name>/ directory should look like:

         ```text
         src/tilegym/transformers/<submodule_name>/
         |- monkey_patch_<compute_name>.py  # Patch function for compute assigned to current subagent.
         |- test_monkey_patch_<compute_name>.py  # Test logic specific to <compute_name> patching.
         |- # Optional [monkey_patch_<other_compute_name>.py, test_monkey_patch_<other_compute_name>.py] pairs created by other subagents assigned with <other_compute_name>s.
         |- modeling_<submodule_name>.py  # Optional if need to modify submodule or function, could be initially created by other subagents.
         ```
      7. Exit the current subagent and let orchestrator agent know that the assigned Transformer compute can be patched by TileGym implementation verified at sub-step 3.2.5 and 3.2.6 and the patch function is available at src/tilegym/transformers/<submodule_name>/monkey_patch_<compute_name>.py.
   3. Aggregate all verified computes and corresponding patches. If none of the compute can be faithfully integrated, exit the workflow and let users know; Otherwise, aggregate all patching logic to a main monkey-patch function `def apply_tilegym_kernel_to_<submodule_name>(...)` and place it at src/tilegym/transformers/monkey_patch.py. Each compute has a corresponding boolean flag as function argument;
   4. Update modeling/transformers/infer.py to include the main monkey-patch function in the inference and benchmark flow. Create a Bash script modeling/transformers/bench_<submodule_name>.sh similar to other bench scripts in that directory. Ensure to use `--use_cutile` at 2nd infer.py call, as we most focus on cuTile backend;
   5. Study [docker-gpu-guide.md](./references/docker-gpu-guide.md), SSH to the node name allocated at sub-step 3.1, build image by modeling/transformers/Dockerfile and verify the end-to-end inference script created at sub-step 3.4. Once all tests pass, release the GPU node.
