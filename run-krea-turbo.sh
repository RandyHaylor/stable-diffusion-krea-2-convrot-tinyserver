#!/usr/bin/env bash
set -euo pipefail

# Set to true to stop the persistent server and release VRAM after this run.
UNLOAD_MODEL_ON_FINISH=false

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${KREA_MODEL_ROOT:-/media/aikenyon/WDRed16TB/models/krea}"
LORA="${KREA_TURBO_LORA:-${MODEL_ROOT}/krea2_raw_to_turbo_r256_LORA/krea2_raw_to_turbo_r256.safetensors}"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 'prompt' [client options]" >&2
    exit 2
fi
if [[ ! -f "$LORA" ]]; then
    echo "missing Turbo LoRA: $LORA" >&2
    exit 1
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
    --steps "${KREA_STEPS:-8}" \
    --cfg "${KREA_CFG:-1.0}" \
    --seed "${KREA_SEED:-0}" \
    --flow-shift "${KREA_FLOW_SHIFT:-1.15}" \
    --lora "$LORA" \
    --lora-strength "${KREA_LORA_STRENGTH:-1.0}" \
    --output-prefix "krea-turbo" \
    --output-dir "${KREA_OUTPUT_DIR:-${ROOT_DIR}/outputs}" \
    "${@:2}")

if [[ "$UNLOAD_MODEL_ON_FINISH" == true ]]; then
    trap '"${ROOT_DIR}/krea-server.sh" stop' EXIT
    "${client[@]}"
else
    exec "${client[@]}"
fi
