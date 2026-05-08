# VSS Alerts Profile — Reference

## Two Modes

| Mode | Flag | How it works | GPU load |
|---|---|---|---|
| **verification** | `-m verification` | CV + behavior analytics generate candidate alerts upstream; VLM reviews each alert clip to reduce false positives | Lower — VLM invoked per alert |
| **real-time** | `-m real-time` | VLM continuously processes live video at periodic intervals; broad coverage without upstream CV dependency | Higher — VLM runs continuously |

## What Gets Deployed

| Service | Purpose |
|---|---|
| NVStreamer | Plays back dataset video to simulate live cameras |
| VIOS | Video ingestion, live streaming, recording, playback |
| RTVI CV | Real-time object detection (Grounding DINO, open-vocabulary) |
| Behavior Analytics | Rule-based alert generation from RTVI CV metadata |
| Alert Verification | VLM-based review of alert video clips |
| Cosmos Reason (NIM) | VLM used by Alert Verification |
| ELK | Log and alert storage |
| VSS Agent | Orchestrates tool calls and queries |
| Nemotron LLM (NIM) | Reasoning and response generation |
| Phoenix | Observability and telemetry |

## GPU Layout (RTXPRO6000BW)

Both GPUs required:

| Device | Role |
|---|---|
| 0 | RT-CV perception (reserved — object detection) |
| 1 | LLM + VLM (`local_shared`) |

## Use Cases

- PPE compliance verification (hard hats, safety vests)
- Restricted area monitoring
- Asset presence/absence detection
- Custom object detection scenarios

## First Run Note

Downloads perception and VLM models from NGC on first run — expect extra time.
