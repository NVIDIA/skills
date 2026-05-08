#!/usr/bin/env bash
# Source this script to populate Brev-specific env vars before
# `docker compose up`. See references/brev.md for details.
#
# Usage:
#   source skills/deploy/scripts/brev_setup.sh          # verbose
#   source skills/deploy/scripts/brev_setup.sh --quiet  # no stdout
#
# Inputs (env vars, all optional):
#   BREV_ENV_FILE       Path to environment file (default /etc/environment).
#                       Override in tests.
#   PROXY_PORT          Nginx proxy port (default 7777).
#   BREV_LINK_PREFIX    Secure-link port prefix (default ${PROXY_PORT}0 per
#                       launchable convention).
#
# Exports (only if BREV_ENV_ID is detected):
#   BREV_ENV_ID         From $BREV_ENV_FILE
#   PROXY_PORT          With default 7777
#   BREV_LINK_PREFIX    Computed secure-link prefix
#
# Exit behavior: this script is sourced — it does not `exit`. Errors are
# reported on stderr and the function returns non-zero; missing
# /etc/environment is not an error (returns 0 silently).

_brev_quiet=""
if [ "${1:-}" = "--quiet" ]; then
    _brev_quiet=1
fi

_brev_log() {
    [ -z "$_brev_quiet" ] && echo "$@"
    return 0
}

_brev_env_file="${BREV_ENV_FILE:-/etc/environment}"

if [ -r "$_brev_env_file" ]; then
    # Strip surrounding quotes and take the first match only.
    _brev_env_id=$(awk -F= '/^BREV_ENV_ID=/ {gsub(/"/, "", $2); print $2; exit}' "$_brev_env_file")
    if [ -n "$_brev_env_id" ]; then
        export BREV_ENV_ID="$_brev_env_id"
    fi
fi

if [ -n "${BREV_ENV_ID:-}" ]; then
    export PROXY_PORT="${PROXY_PORT:-7777}"
    # Brev Launchable convention: secure-link prefix is ${PROXY_PORT}0 (e.g.
    # 7777 → 77770).  For manually-created links, override with
    # BREV_LINK_PREFIX=<prefix> before sourcing.
    export BREV_LINK_PREFIX="${BREV_LINK_PREFIX:-${PROXY_PORT}0}"
    _brev_base_url="https://${BREV_LINK_PREFIX}-${BREV_ENV_ID}.brevlab.com"
    _brev_log "Brev detected:"
    _brev_log "  BREV_ENV_ID      = $BREV_ENV_ID"
    _brev_log "  PROXY_PORT       = $PROXY_PORT"
    _brev_log "  BREV_LINK_PREFIX = $BREV_LINK_PREFIX"
    _brev_log "  UI URL           = $_brev_base_url"
    unset _brev_base_url
else
    _brev_log "No BREV_ENV_ID in $_brev_env_file — not a Brev instance (or env file missing)."
fi

unset _brev_quiet _brev_log _brev_env_file _brev_env_id
