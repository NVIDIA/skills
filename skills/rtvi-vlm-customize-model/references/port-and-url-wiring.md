# RTVI-VLM Port and Base-URL Wiring

Four variables must name the same port before a local Alerts stack comes up:

| Variable | Source | Consumed by |
|---|---|---|
| `RTVI_VLM_PORT` | profile `.env`, passed through untouched | `rtvi-vlm` Compose port mapping (no default) |
| `RTVI_VLM_BASE_URL` | profile `.env`, passed through untouched | `vss-agent` with `VLM_MODEL_TYPE=rtvi` |
| `RTVI_VLM_ENDPOINT` | profile `.env`, passed through untouched | `rtvi-vlm` itself, as `VIA_VLM_ENDPOINT` |
| `VLM_BASE_URL` | written by `dev-profile.sh` | `alert-bridge` (vlm-as-verifier), `vss-agent` `nim` / `openai` / `vllm` profiles |

They do not share a source. For the `alerts` and `lvs` profiles with a local
VLM, `dev-profile.sh` writes `VLM_BASE_URL` as a hardcoded
`http://<host-ip>:8018` on every run, overwriting whatever the profile `.env`
contains. The literal `8018` is not derived from `RTVI_VLM_PORT`, so the four
agree only at the default port.

The `--vlm-base-url` flag (`VLM_ENDPOINT_URL`) is not an alternative: it applies
only with `--use-remote-vlm`, and the local `alerts` / `lvs` branch overwrites
`VLM_BASE_URL` afterwards.

## Default port (8018)

Add `RTVI_VLM_PORT` and parameterize the two hardcoded `8018` lines in the
Alerts profile `.env`, then rerun `dev-profile.sh`. Edit `.env`, not
`generated.env` — the generator overwrites it.

```bash
# ${VSS_PROFILE_DIR}/.env
RTVI_VLM_PORT=8018                                        # add this line
RTVI_VLM_BASE_URL=http://${HOST_IP}:${RTVI_VLM_PORT}     # update: was http://${HOST_IP}:8018
RTVI_VLM_ENDPOINT=http://${HOST_IP}:${RTVI_VLM_PORT}/v1  # update: was http://${HOST_IP}:8018/v1
```

## Any other port

Set `RTVI_VLM_PORT` in `.env` as above, then repair `VLM_BASE_URL` in
`generated.env` after every `dev-profile.sh` run. Patch only the port so the
host IP the generator resolved is preserved:

```bash
GENERATED_ENV="${VSS_PROFILE_DIR}/generated.env"
PORT="$(sed -n 's|^RTVI_VLM_PORT=||p' "${GENERATED_ENV}" | tr -d "\"' ")"
test -n "${PORT}" || { echo "RTVI_VLM_PORT not found in ${GENERATED_ENV}" >&2; exit 1; }

sed -i -E "s|^(VLM_BASE_URL=http://[^:]+):[0-9]+|\1:${PORT}|" "${GENERATED_ENV}"

grep -E '^(RTVI_VLM_PORT|RTVI_VLM_BASE_URL|RTVI_VLM_ENDPOINT|VLM_BASE_URL)=' \
  "${GENERATED_ENV}"
```

Skipping this patch publishes RTVI-VLM where `alert-bridge` and the non-`rtvi`
`vss-agent` profiles are not looking: they keep dialing 8018.

Set `RTVI_VLM_BASE_URL` by hand when pointing `vss-agent` at an RTVI-VLM
instance on a different host.
