# NVIDIA-AI-IOT GHCR images (Jetson-oriented)

Browse versions and tags at the org registry: [NVIDIA-AI-IOT packages](https://github.com/orgs/NVIDIA-AI-IOT/packages).

Commonly used **image families** for GenAI on Jetson (confirm the latest tag on the package page):

| Package / image family | Typical use |
|------------------------|-------------|
| `vllm` | `latest-jetson-orin` for Orin LLM/VLM serving before JetPack 7.2 / L4T r39; use upstream/native vLLM 0.20+ on Thor and Orin r39+ |
| `llama_cpp` | GGUF inference with GPU offload |
| `ollama` | Quick local GGUF chat |
| `live-vlm-webui` | Browser UI for VLMs |

Pick **Orin** vs **Thor** from **`JETSON_GENERATION`** after running `skills/jetson-diagnostic/scripts/detect_jetson.sh`. For vLLM, Orin JetPack 7.2 / L4T r39+ and Thor use upstream/native vLLM 0.20+; older Orin maps to GHCR `latest-jetson-orin`. **`JETSON_SKU`** remains the legacy bucket; **`JETSON_PRODUCT_LINE`** refines Orin to `orin-agx` / `orin-nx` / `orin-nano` and Thor today to **`thor-agx`**.
