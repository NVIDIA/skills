#!/usr/bin/env bash
# preflight.sh — Probe local system for Proteina-Complexa readiness; emit JSON.
#
# Probes GPU (name/VRAM/count/driver/CUDA), disk free in $CKPT_PATH, the six
# canonical Complexa ckpts, the six tool binaries (foldseek/mmseqs/dssp/hbplus/
# sc/rf3), dotenv readability + required-var presence, community model paths
# (AF2_DIR/ESM_DIR/RF3_CKPT_PATH), and git SHA. Every probe degrades to
# {available:false} / {exists:false} rather than failing.
#
# Usage: bash preflight.sh [--quiet] [--out PATH] [--help]
set -euo pipefail

QUIET=0; OUT="./complexa_setup/preflight.json"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quiet) QUIET=1; shift ;;
        --out)   OUT="${2:-}"; shift 2 ;;
        --help|-h) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

json_str() {
    local v="${1-}"
    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import json,sys; sys.stdout.write(json.dumps(sys.argv[1]))' "$v"
    else
        printf '"%s"' "$(sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' <<<"$v")"
    fi
}

# Parse the repository dotenv as data and dump only the allowlisted variables as
# KEY<TAB>VALUE. Never source it: dotenv files are configuration, not shell code.
ENV_FILE="$PWD/.env"; ENV_LOADED=false
declare -A V=()
if [[ -f "$ENV_FILE" ]]; then
    ENV_LOADED=true
    DUMP=$(python3 - "$ENV_FILE" <<'PY' 2>/dev/null || true
import os
import re
import shlex
import sys
from pathlib import Path

requested = (
    "LOCAL_CODE_PATH", "LOCAL_DATA_PATH", "CKPT_PATH", "LOCAL_CHECKPOINT_PATH",
    "COMPLEXA_INIT", "FOLDSEEK_EXEC", "MMSEQS_EXEC", "DSSP_EXEC",
    "HBPLUS_EXEC", "SC_EXEC", "RF3_EXEC_PATH", "AF2_DIR", "ESM_DIR",
    "RF3_CKPT_PATH",
)
assignment = re.compile(r"(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\Z")
reference = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
values: dict[str, str] = {}

for raw_line in Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    match = assignment.fullmatch(raw_line.strip())
    if match is None:
        continue
    key, raw_value = match.groups()
    lexer = shlex.shlex(raw_value, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        value = " ".join(lexer)
    except ValueError:
        continue
    values[key] = value

def resolve(value: str) -> str:
    for _ in range(len(values) + 1):
        expanded = reference.sub(
            lambda match: values.get(match.group(1), os.environ.get(match.group(1), match.group(0))),
            value,
        )
        if expanded == value:
            return expanded
        value = expanded
    return value

for key in requested:
    print(f"{key}\t{resolve(values.get(key, ''))}")
PY
    )
    while IFS=$'\t' read -r k v; do [[ -n "$k" ]] && V["$k"]="$v"; done <<<"$DUMP"
fi
# Fall through to live env for any unset keys
for k in LOCAL_CODE_PATH LOCAL_DATA_PATH CKPT_PATH LOCAL_CHECKPOINT_PATH \
         COMPLEXA_INIT FOLDSEEK_EXEC MMSEQS_EXEC DSSP_EXEC HBPLUS_EXEC \
         SC_EXEC RF3_EXEC_PATH AF2_DIR ESM_DIR RF3_CKPT_PATH; do
    [[ -z "${V[$k]:-}" ]] && V["$k"]="${!k-}"
done

# Resolve CKPT_PATH: explicit -> LOCAL_CHECKPOINT_PATH -> $LOCAL_CODE_PATH/ckpts
if [[ -z "${V[CKPT_PATH]:-}" ]]; then
    if   [[ -n "${V[LOCAL_CHECKPOINT_PATH]:-}" ]]; then V[CKPT_PATH]="${V[LOCAL_CHECKPOINT_PATH]}"
    elif [[ -n "${V[LOCAL_CODE_PATH]:-}"       ]]; then V[CKPT_PATH]="${V[LOCAL_CODE_PATH]}/ckpts"
    fi
fi

MISSING=()
for req in LOCAL_CODE_PATH LOCAL_DATA_PATH CKPT_PATH; do
    [[ -z "${V[$req]:-}" ]] && MISSING+=("$req")
done

# ---- GPU ----
GPU_JSON='{"available":false}'
if command -v nvidia-smi >/dev/null 2>&1; then
    OUT_LINES=$(nvidia-smi --query-gpu=name,memory.total,driver_version \
                           --format=csv,noheader,nounits 2>/dev/null || true)
    if [[ -n "$OUT_LINES" ]]; then
        N=$(wc -l <<<"$OUT_LINES" | tr -d ' ')
        F=$(head -n1 <<<"$OUT_LINES")
        NAME=$(awk -F',' '{gsub(/^ +| +$/,"",$1); print $1}' <<<"$F")
        VRAM_MIB=$(awk -F',' '{gsub(/[^0-9]/,"",$2); print $2}' <<<"$F")
        VRAM_GB=$(( VRAM_MIB / 1024 ))
        DRV=$(awk -F',' '{gsub(/^ +| +$/,"",$3); print $3}' <<<"$F")
        CUDA=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n1)
        printf -v GPU_JSON '{"available":true,"name":%s,"vram_gb":%s,"count":%s,"driver":%s,"cuda":%s}' \
            "$(json_str "$NAME")" "$VRAM_GB" "$N" "$(json_str "$DRV")" "$(json_str "${CUDA:-unknown}")"
    fi
fi

# ---- Disk ----
DISK_FREE="null"; DISK_TARGET="${V[CKPT_PATH]:-}"
if [[ -n "$DISK_TARGET" ]]; then
    DP="$DISK_TARGET"; [[ ! -d "$DP" ]] && DP="$(dirname -- "$DP" 2>/dev/null || echo /)"
    if [[ -d "$DP" ]]; then
        FREE_KB=$(df -P -k -- "$DP" 2>/dev/null | awk 'NR==2{print $4}')
        [[ -n "${FREE_KB:-}" ]] && DISK_FREE=$(( FREE_KB / 1024 / 1024 ))
    fi
fi
printf -v DISK_JSON '{"ckpt_path":%s,"free_gb":%s}' "$(json_str "$DISK_TARGET")" "$DISK_FREE"

# ---- Checkpoints ----
CKPT_ITEMS=()
for name in complexa.ckpt complexa_ae.ckpt complexa_ligand.ckpt complexa_ligand_ae.ckpt complexa_ame.ckpt complexa_ame_ae.ckpt; do
    p="${V[CKPT_PATH]%/}/$name"; ex=false; size="null"; sha="null"
    if [[ -n "${V[CKPT_PATH]:-}" && -f "$p" ]]; then
        ex=true
        sz=$(stat -c %s -- "$p" 2>/dev/null || stat -f %z -- "$p" 2>/dev/null || echo "")
        [[ -n "$sz" ]] && size="$sz"
        if command -v sha256sum >/dev/null 2>&1; then
            s=$(sha256sum -- "$p" 2>/dev/null | cut -c1-16 || echo "")
        elif command -v shasum >/dev/null 2>&1; then
            s=$(shasum -a 256 -- "$p" 2>/dev/null | cut -c1-16 || echo "")
        fi
        [[ -n "${s:-}" ]] && sha="$(json_str "$s")"
    fi
    printf -v item '%s:{"path":%s,"exists":%s,"size":%s,"sha256":%s}' \
        "$(json_str "$name")" "$(json_str "$p")" "$ex" "$size" "$sha"
    CKPT_ITEMS+=("$item")
done
CKPT_JSON="{$(IFS=,; echo "${CKPT_ITEMS[*]}")}"

# ---- Tools ----
TOOL_ITEMS=()
for entry in "foldseek=${V[FOLDSEEK_EXEC]:-}" "mmseqs=${V[MMSEQS_EXEC]:-}" \
             "dssp=${V[DSSP_EXEC]:-}"         "hbplus=${V[HBPLUS_EXEC]:-}" \
             "sc=${V[SC_EXEC]:-}"             "rf3=${V[RF3_EXEC_PATH]:-}"; do
    k="${entry%%=*}"; p="${entry#*=}"; ex=false
    [[ -n "$p" && ( -x "$p" || -f "$p" ) ]] && ex=true
    printf -v item '%s:{"path":%s,"exists":%s}' "$(json_str "$k")" "$(json_str "$p")" "$ex"
    TOOL_ITEMS+=("$item")
done
TOOLS_JSON="{$(IFS=,; echo "${TOOL_ITEMS[*]}")}"

# ---- Community models ----
cm() { local k="$1" p="$2" ex=false; [[ -n "$p" && -e "$p" ]] && ex=true; printf '%s:{"path":%s,"exists":%s}' "$(json_str "$k")" "$(json_str "$p")" "$ex"; }
COMMUNITY_JSON="{$(cm AF2_DIR "${V[AF2_DIR]:-}"),$(cm ESM_DIR "${V[ESM_DIR]:-}"),$(cm RF3_CKPT_PATH "${V[RF3_CKPT_PATH]:-}")}"

# ---- Env summary ----
MISS_JSON="[]"
if [[ ${#MISSING[@]} -gt 0 ]]; then
    parts=(); for m in "${MISSING[@]}"; do parts+=("$(json_str "$m")"); done
    MISS_JSON="[$(IFS=,; echo "${parts[*]}")]"
fi
printf -v ENV_JSON '{".env_loaded":%s,".env_path":%s,"missing_required":%s,"LOCAL_CODE_PATH":%s,"LOCAL_DATA_PATH":%s,"CKPT_PATH":%s}' \
    "$ENV_LOADED" "$(json_str "$ENV_FILE")" "$MISS_JSON" \
    "$(json_str "${V[LOCAL_CODE_PATH]:-}")" "$(json_str "${V[LOCAL_DATA_PATH]:-}")" "$(json_str "${V[CKPT_PATH]:-}")"

# ---- Git SHA ----
GIT_SHA="unknown"; GIT_DIR="${V[LOCAL_CODE_PATH]:-$PWD}"
if command -v git >/dev/null 2>&1; then
    c=$(git -C "$GIT_DIR" rev-parse --short HEAD 2>/dev/null || true)
    [[ -z "$c" ]] && c=$(git rev-parse --short HEAD 2>/dev/null || true)
    [[ -n "$c" ]] && GIT_SHA="$c"
fi

TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
printf -v DOC '{"timestamp":%s,"gpu":%s,"disk":%s,"checkpoints":%s,"tools":%s,"env":%s,"community_models":%s,"complexa_runtime":%s,"git_sha":%s}' \
    "$(json_str "$TS")" "$GPU_JSON" "$DISK_JSON" "$CKPT_JSON" "$TOOLS_JSON" "$ENV_JSON" "$COMMUNITY_JSON" \
    "$(json_str "${V[COMPLEXA_INIT]:-}")" "$(json_str "$GIT_SHA")"

PRETTY="$DOC"
if command -v jq >/dev/null 2>&1; then
    PRETTY=$(jq . <<<"$DOC" 2>/dev/null || printf '%s' "$DOC")
elif command -v python3 >/dev/null 2>&1; then
    PRETTY=$(python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin), indent=2))' <<<"$DOC" 2>/dev/null || printf '%s' "$DOC")
fi

OUT_PARENT=$(dirname -- "$OUT")
[[ "$OUT_PARENT" == "." ]] || mkdir -p -- "$OUT_PARENT"
OUT_TMP=$(mktemp "${OUT}.tmp.XXXXXX")
trap 'rm -f -- "$OUT_TMP"' EXIT
printf '%s\n' "$PRETTY" > "$OUT_TMP"
mv -- "$OUT_TMP" "$OUT"
trap - EXIT
[[ "$QUIET" == "1" ]] || printf '%s\n' "$PRETTY"
