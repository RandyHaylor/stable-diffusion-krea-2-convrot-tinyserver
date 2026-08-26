#!/usr/bin/env bash
set -euo pipefail

# Set to true to stop the persistent server and release VRAM after this run.
UNLOAD_MODEL_ON_FINISH=false
# Set to false for quiet operation. When true, model-load and sampling progress
# from the persistent server is streamed into this terminal.
SHOW_SERVER_LOG=true

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 'prompt' [client options]" >&2
    exit 2
fi

if ! "${ROOT_DIR}/krea-server.sh" status >/dev/null 2>&1; then
    "${ROOT_DIR}/krea-server.sh" start
    for _ in {1..240}; do
        "${ROOT_DIR}/krea-server.sh" status 2>/dev/null | grep -q '^ready' && break
        sleep 1
    done
fi
if ! "${ROOT_DIR}/krea-server.sh" status 2>/dev/null | grep -q '^ready'; then
    echo "Krea server did not become ready; see: ${ROOT_DIR}/.runtime/krea-server.log" >&2
    exit 1
fi

client=(python3 "${ROOT_DIR}/scripts/krea_server_client.py" \
    "$1" \
    --host "${KREA_SERVER_HOST:-127.0.0.1}" \
    --port "${KREA_SERVER_PORT:-1234}" \
    --width "${KREA_WIDTH:-1024}" \
    --height "${KREA_HEIGHT:-1024}" \
    --steps "${KREA_STEPS:-52}" \
    --cfg "${KREA_CFG:-3.5}" \
    --seed "${KREA_SEED:-0}" \
    --output-prefix "krea-raw" \
    --output-dir "${KREA_OUTPUT_DIR:-${ROOT_DIR}/outputs}" \
    "${@:2}")

log_follow_pid=""
cleanup() {
    if [[ -n "$log_follow_pid" ]]; then
        kill "$log_follow_pid" 2>/dev/null || true
        wait "$log_follow_pid" 2>/dev/null || true
    fi
    if [[ "$UNLOAD_MODEL_ON_FINISH" == true ]]; then
        "${ROOT_DIR}/krea-server.sh" stop || true
    fi
}
trap cleanup EXIT INT TERM

if [[ "$SHOW_SERVER_LOG" == true ]]; then
    echo "Following server progress (the first request loads model weights)..."
    tail -n 0 -f "${ROOT_DIR}/.runtime/krea-server.log" &
    log_follow_pid=$!
fi
"${client[@]}"
