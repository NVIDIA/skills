# PAS Inference Endpoints


## Table of Contents

- [Required Endpoints](#required-endpoints)
- [Option A: Reuse Existing In-Cluster NIMs (default)](#option-a-reuse-existing-in-cluster-nims-default)
- [Option B: Deploy Image-Edit NIM](#option-b-deploy-image-edit-nim)
- [Verify Endpoint Health Before Submitting](#verify-endpoint-health-before-submitting)
- [Option C: External Endpoint Override (opt-in only)](#option-c-external-endpoint-override-opt-in-only)

## Required Endpoints

PAS workflows call three OpenAI-compatible endpoints:

| Role | Default NIM service | Default URL |
|------|-------------------|-------------|
| Image Edit | `qwen-image-edit-2511` | `http://qwen-image-edit-2511.osmo-nims.svc.cluster.local:8000/v1` |
| VLM (verification + captioning) | `qwen3-vl` (shared with VDA) | `http://qwen3-vl.osmo-nims.svc.cluster.local:8000/v1` |
| LLM (MCQ generation) | `qwen25-14b` (shared with VDA) | `http://qwen25-14b.osmo-nims.svc.cluster.local:8000/v1` |

## Option A: Reuse Existing In-Cluster NIMs (default)

VLM (`qwen3-vl`) and LLM (`qwen25-14b`) are the same services used by VDA.
If VDA endpoints are already healthy, PAS reuses them directly.

For Image Edit, PAS uses `Qwen/Qwen-Image-Edit-2511` (the generic upstream
checkpoint) served by `vllm/vllm-omni` with `vllm serve ... --omni`. This is
different from DIG which uses the finetuned
`nvidia/Qwen-Image-Edit-NVPCB-OVSL2SL` variant.

> **Image-edit endpoint must be vLLM-Omni, not the Triton NIM.** The worker calls
> `/v1/chat/completions`. The Triton-based `nvcr.io/nim/qwen/qwen-image-edit`
> serves only `/v1/infer` + `/v1/images/edits`, so it returns `404` and produces
> no images. Always deploy via the manifest below (it sets the required `--omni`
> flag).

## Option B: Deploy Image-Edit NIM

If the `qwen-image-edit-2511` NIM is not already running, deploy it via the
checked-in NIMService manifest:

```bash
kubectl create namespace osmo-nims --dry-run=client -o yaml | kubectl apply -f -
kubectl -n osmo-nims create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="${HF_TOKEN}" \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f references/nim/qwen-image-edit-2511.yaml
kubectl -n osmo-nims wait --for=condition=Ready \
  nimservice.apps.nvidia.com/qwen-image-edit-2511 --timeout=60m
```

For VLM and LLM, use the VDA deployment path:

```bash
export NIM_SERVICES="qwen3-vl qwen25-14b"
skills/physical-ai-infrastructure-setup-and-resilient-scaling/components/inference-nim-operator/scripts/install.sh
```

## Verify Endpoint Health Before Submitting

```bash
# Image Edit
kubectl run curl-ie -n osmo-nims --rm -i --restart=Never \
  --image=curlimages/curl -- \
  curl -fsS http://qwen-image-edit-2511.osmo-nims.svc.cluster.local:8000/v1/models

# VLM
kubectl run curl-vlm -n osmo-nims --rm -i --restart=Never \
  --image=curlimages/curl -- \
  curl -fsS http://qwen3-vl.osmo-nims.svc.cluster.local:8000/v1/models

# LLM
kubectl run curl-llm -n osmo-nims --rm -i --restart=Never \
  --image=curlimages/curl -- \
  curl -fsS http://qwen25-14b.osmo-nims.svc.cluster.local:8000/v1/models
```

Proceed only when all three return healthy model lists.

## Option C: External Endpoint Override (opt-in only)

Use external endpoints when explicitly requested:

```bash
osmo workflow submit assets/configs/osmo/e2e.yaml \
  --set-string image_edit_url=https://<host>/v1 \
               vlm_url=https://<host>/v1 \
               llm_url=https://<host>/v1
```

Hosted equivalents on build.nvidia.com:
- https://build.nvidia.com/qwen/qwen-image-edit
- https://build.nvidia.com/qwen/qwen3-vl-30b-a3b-instruct
- https://build.nvidia.com/qwen/qwen2.5-14b-instruct

## Model safety disclaimer

Users are responsible for model inputs and outputs. Users are responsible for
ensuring safe integration of this model, including implementing guardrails as
well as other safety mechanisms, prior to deployment.
