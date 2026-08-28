#!/usr/bin/env bash
set -euo pipefail

# Set to true to stop the persistent server and release VRAM after this run.
UNLOAD_MODEL_ON_FINISH=false
# Set to false for quiet operation. When true, model-load and sampling progress
# from the persistent server is streamed into this terminal.
SHOW_SERVER_LOG=true

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${KREA_MODEL_ROOT:-${ROOT_DIR}/models}"
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
    --width "${KREA_WIDTH:-1792}" \
    --height "${KREA_HEIGHT:-1792}" \
    --steps "${KREA_STEPS:-8}" \
    --cfg "${KREA_CFG:-1.0}" \
    --seed "${KREA_SEED:-0}" \
    --sampler "${KREA_SAMPLER:-res_2s}" \
    --scheduler "${KREA_SCHEDULER:-beta}" \
    --extra-sample-args "${KREA_EXTRA_SAMPLE_ARGS:-alpha=0.5,beta=0.7}" \
    --flow-shift "${KREA_FLOW_SHIFT:-1.15}" \
    --hires-width "${KREA_HIRES_WIDTH:-0}" \
    --hires-height "${KREA_HIRES_HEIGHT:-0}" \
    --hires-steps "${KREA_HIRES_STEPS:-3}" \
    --hires-denoise "${KREA_HIRES_DENOISE:-0.5}" \
    --hires-upscaler "${KREA_HIRES_UPSCALER:-Latent}" \
    --lora "$LORA" \
    --lora-strength "${KREA_LORA_STRENGTH:-0.6}" \
    --output-prefix "${KREA_OUTPUT_PREFIX:-krea-turbo}" \
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
    echo "Following server progress (the first request loads model weights and the Turbo LoRA)..."
    tail -n 0 -f "${ROOT_DIR}/.runtime/krea-server.log" &
    log_follow_pid=$!
fi
"${client[@]}"
