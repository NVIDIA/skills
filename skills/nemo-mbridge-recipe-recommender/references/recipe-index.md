# Recipe Index (Library & Benchmark)

Full per-family recipe tables for the `nemo-mbridge-recipe-recommender` skill.
See [`../SKILL.md`](../SKILL.md) for how to choose among these (the decision
tree, parallelism resizing rules, and pitfalls live there).

## Library Recipe Index

All recipes live under `src/megatron/bridge/recipes/`. Each function returns a
`ConfigContainer` with model, training, optimizer, and data settings.

### Llama

| Recipe | Mode | TP | PP | CP | SP | GPUs (min) | Seq Len |
|--------|------|----|----|----|----|------------|---------|
| `llama2_7b_pretrain_config` | Pretrain | 2 | 1 | — | — | 2 | 4K |
| `llama3_8b_pretrain_config` | Pretrain | 2 | 1 | — | ✓ | 2 | 8K |
| `llama3_8b_16k_pretrain_config` | Pretrain | 2 | 1 | 2 | ✓ | 4 | 16K |
| `llama3_8b_64k_pretrain_config` | Pretrain | 2 | 1 | 4 | ✓ | 8 | 64K |
| `llama3_8b_128k_pretrain_config` | Pretrain | 2 | 1 | 8 | ✓ | 16 | 128K |
| `llama3_70b_pretrain_config` | Pretrain | 8 | 4 | — | ✓ | 32 | 8K |
| `llama3_70b_16k_pretrain_config` | Pretrain | 8 | 4 | 2 | ✓ | 64 | 16K |
| `llama3_70b_64k_pretrain_config` | Pretrain | 8 | 4 | 4 | ✓ | 128 | 64K |
| `llama31_405b_pretrain_config` | Pretrain | 8 | 16 | — | ✓ | 128 | 8K |
| `llama3_8b_sft_config` | SFT | 2 | 1 | — | ✓ | 2 | 8K |
| `llama3_70b_sft_config` | SFT | 4 | 4 | — | ✓ | 16 | 8K |
| `llama31_405b_sft_config` | SFT | 8 | 8 | — | ✓ | 64 | 8K |
| `llama3_8b_peft_config` | PEFT | 1 | 1 | — | — | 1 | 8K |
| `llama3_70b_peft_config` | PEFT | 2 | 4 | — | ✓ | 8 | 8K |
| `llama31_405b_peft_config` | PEFT | 4 | 8 | — | ✓ | 32 | 8K |

### Qwen2 / Qwen2.5

| Recipe | Mode | TP | PP | Sizes |
|--------|------|----|----|-------|
| `qwen2_*_{pretrain,sft,peft}_config` | All | 1–8 | 1–4 | 500M, 1.5B, 7B, 14B, 32B, 72B |
| `qwen25_*_{pretrain,sft,peft}_config` | All | 1–8 | 1–4 | 500M, 1.5B, 3B, 7B, 14B, 32B, 72B |

### Qwen3 (Dense)

| Recipe | Mode | TP | PP | CP | GPUs | Sizes / notes |
|--------|------|----|----|----|------|---------------|
| `qwen3_8b_pretrain_config` | Pretrain | 1 | 1 | — | 16 | Bounded convergence cohort |
| `qwen3_8b_sft_config` | SFT | 4 | 1 | — | 4 | 2K bounded convergence cohort |
| `qwen3_8b_sft_32k_config` | SFT | 4 | 1 | 2 | 8 | Separate 32K long-context cohort |
| `qwen3_8b_peft_config` | PEFT | 1 | 1 | — | 1 | Bounded LoRA/DoRA cohort |
| `qwen3_*_{pretrain,sft,peft}_config` | All | 1–8 | 1–2 | — | varies | Other dense sizes, 600M–32B |
| `qwen3_600m_sft_128k_config` | SFT | 1 | 1 | 8 | 8 | 600M, 128K sequence |

### Qwen3 MoE

| Recipe | Mode | TP | PP | EP | CP | GPUs |
|--------|------|----|----|----|----|------|
| `qwen3_30b_a3b_pretrain_config` | Pretrain | 1 | 1 | 16 | — | 16 |
| `qwen3_30b_a3b_sft_config` | SFT | 1 | 1 | 16 | — | 16 |
| `qwen3_30b_a3b_peft_config` | PEFT | 4 | 1 | 4 | — | 4 |
| `qwen3_235b_a22b_pretrain_config` | Pretrain | 4 | 16 | 8 | 2 | 256 |
| `qwen3_235b_a22b_sft_config` | SFT | 4 | 16 | 4 | — | 64 |
| `qwen3_235b_a22b_peft_config` | PEFT | 1 | 4 | 4 | — | 16 |

### Qwen3-Next

| Recipe | Mode | TP | PP | EP |
|--------|------|----|----|-----|
| `qwen3_next_80b_a3b_pretrain_config` | Pretrain | 1 | 4 | 8 |
| `qwen3_next_80b_a3b_sft_config` | SFT | 1 | 2 | 8 |
| `qwen3_next_80b_a3b_peft_config` | PEFT | 1 | 1 | 4 |

### DeepSeek

| Recipe | Mode | TP | PP | EP | GPUs |
|--------|------|----|----|-----|------|
| `deepseek_v2_lite_pretrain_config` | Pretrain | 1 | 1 | 8 | 8 |
| `deepseek_v2_pretrain_config` | Pretrain | 1 | 4 | 32 | 128 |
| `deepseek_v3_pretrain_config` | Pretrain | 2 | 16 | 64 | 2048 |
| `deepseek_v3_pretrain_config_32nodes` | Pretrain | 2 | 8 | 32 | 256 |

### GLM-4.5

| Recipe | Mode | TP | PP | EP | GPUs |
|--------|------|----|----|-----|------|
| `glm45_355b_pretrain_config` | Pretrain | 2 | 8 | 16 | 256 |
| `glm45_air_106b_pretrain_config` | Pretrain | 1 | 4 | 8 | 32 |
| `glm45_355b_sft_config` | SFT | 2 | 8 | 16 | 256 |
| `glm45_air_106b_sft_config` | SFT | 1 | 4 | 8 | 32 |
| `glm45_355b_peft_config` | PEFT | 2 | 4 | 4 | 32 |
| `glm45_air_106b_peft_config` | PEFT | 1 | 2 | 4 | 8 |

### Gemma

| Recipe | Mode | TP | PP | Sizes |
|--------|------|----|----|-------|
| `gemma2_*_{pretrain,sft,peft}_config` | All | 2–8 | 1–2 | 2B, 9B, 27B |
| `gemma3_1b_{pretrain,sft,peft}_config` | All | 1 | 1 | 1B (32K seq) |

### NemotronH / Nemotron

| Recipe | Mode | TP | PP | EP | Notes |
|--------|------|----|----|-----|-------|
| `nemotronh_{4b,8b,47b,56b}_*_config` | P/S/PEFT | 1–8 | 1–4 | — | Dense SSM-hybrid |
| `nemotron_3_nano_*_config` | P/S/PEFT | varies | 1 | 8 | MoE + Mamba |
| `nemotron_3_super_*_config` | P/S/PEFT | 4 | 1 | 8 | MoE + Mamba, ~40% CUDA graph gain |
| `nemotron_nano_{9b,12b}_v2_*_config` | P/S/PEFT | varies | 1 | — | Dense |

### Other Models

| Recipe | Mode | Notes |
|--------|------|-------|
| `moonlight_16b_pretrain_config` | Pretrain | 16 GPUs, TP1/PP1/EP8 with HybridEP; GBS/MBS 1024/2 bounded convergence cohort |
| `moonlight_16b_sft_config` | SFT | 8 GPUs, TP1/PP1/EP8; 8K offline packing, GBS/MBS 8/1, 65,536 tokens/update |
| `moonlight_16b_peft_config` | PEFT | 4 GPUs, TP1/PP1/EP4; bounded LoRA/DoRA cohort |
| `moonlight_16b_sft_8k_config` | SFT | 8 GPUs, TP2/PP1/CP2/EP8; separate 8K cohort |
| `olmoe_7b_{pretrain,sft,peft}_config` | All | MoE EP=8 |
| `ministral3_{3b,8b,14b}_{sft,peft}_config` | SFT/PEFT | Dense |
| `gpt_oss_20b_*_config` | All | MoE + FP8/MXFP8 variants |
| `gpt_oss_120b_*_config` | All | MoE |
| `vanilla_gpt_pretrain_config` | Pretrain | MLM/Bridge parity baseline |
| `gpt3_175b_pretrain_config` | Pretrain | TP=4, PP=8, VP=6 |
| `kimi_k2_pretrain_config` | Pretrain | 1T MoE, TP=2 PP=16 EP=32 |

### VLM Recipes

| Recipe | Mode | TP | PP | EP | GPUs |
|--------|------|----|----|-----|------|
| `gemma3_vl_{4b,12b,27b}_{sft,peft}_config` | SFT/PEFT | 1–8 | 1–2 | — | 1–16 |
| `qwen25_vl_{3b,7b,32b,72b}_{sft,peft}_config` | SFT/PEFT | 1–8 | 1–4 | — | 1–32 |
| `qwen3_vl_{8b,30b_a3b,235b_a22b}_{sft,peft}_config` | SFT/PEFT | 1–4 | 1–8 | 1–32 | 1–512 |
| `qwen35_vl_*_{sft,peft}_config` | SFT/PEFT | varies | varies | varies | varies |
| `glm_45v_{sft,peft}_config` | SFT/PEFT | 1 | 8 | 4–16 | 64–512 |
| `nemotron_nano_v2_vl_12b_{sft,peft}_config` | SFT/PEFT | 2–4 | 1 | — | 8 |

### Diffusion Recipes

| Recipe | Mode | TP | CP |
|--------|------|----|----|
| `wan_1_3B_{pretrain,sft}_config` | P/SFT | 1 | 8 |
| `wan_14B_{pretrain,sft}_config` | P/SFT | 2 | 4 |
| `flux_12b_{pretrain,sft}_config` | P/SFT | 2 | 1 |

---

## Benchmark Recipe Index

Benchmark recipe source lives under `src/megatron/bridge/perf_recipes/`. The
compatibility launcher in `scripts/performance/` resolves those flat recipe
names and derives compatibility workload views from the selected flat recipe
when legacy helper paths still need them.

> **Important:** Benchmark recipes are designed for **upper-bound throughput
> measurements**, not production training. Text benchmarks use mock data by
> default; Qwen-VL and Wan retain their model-specific datasets. Throughput
> numbers are aspirational targets, not validated convergence configs.

### Llama 3 / 3.1

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| Llama 3 8B | 8 | H100, B200, B300, GB200, GB300, R100 | CUDA graphs (local), FSDP on GB variants |
| Llama 3 70B | 64 | H100, B200, B300, GB200, GB300 | TP comm overlap (userbuffers), FSDP, CUDA graphs |
| Llama 3.1 405B | 128–1024 | H100, B200, B300, GB200, GB300 | TP+CP comm overlap (userbuffers), FSDP, heavy PP/VP |

SFT/LoRA variants also exist (e.g. 8B SFT with packed sequences, 70B SFT on 32 GPUs).

### DeepSeek V3

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| DeepSeek V3 (671B MoE) | 256–1024 | H100, B200, B300, GB200, GB300 | HybridEP dispatcher, MLA recompute, CUDA graphs (TE scoped) |

### Qwen3 MoE

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| Qwen3 30B-A3B | 8–16 | H100, B200, B300, GB200, GB300 | MoE alltoall/flex dispatcher |
| Qwen3 235B-A22B | 64–256 | H100, B200, B300, GB200, GB300 | TP comm overlap, CUDA graphs, MoE a2a overlap |
| Qwen3-Next 80B-A3B | 64–128 | H100, B200, B300, GB200, GB300 | EP 64–128 |

### Qwen3-VL

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| Qwen3-VL 30B-A3B | 8–16 | H100, B200, B300, GB200, GB300 | VLM + MoE |
| Qwen3-VL 235B-A22B | 64–256 | H100, B200, B300, GB200, GB300 | VLM + MoE, TP comm overlap |

### Kimi K2

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| Kimi K2 (1T MoE) | 256–1024 | H100, B200, B300, GB200, GB300 | Muon/Adam optimizer, HybridEP, pipeline layout helpers |

### NemotronH

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| Nemotron 3 Nano (30B MoE+Mamba) | 8–16 | H100, B200, B300, GB200, GB300 | TE CUDA graphs (attn+mamba+moe), HybridEP |
| Nemotron 3 Super | 64 | H100, B200, B300, GB200, GB300 | TE CUDA graphs, EP=64 |
| NemotronH 56B | 64 | H100, B200, B300 | TP=2–8, TE graphs (mamba+attn) |

### GPT-OSS

| Model | GPUs | GPU Types | Key Features |
|-------|------|-----------|--------------|
| GPT-OSS 120B | 64 | H100, B200, GB200 | EP=64, HybridEP on GB200 |
