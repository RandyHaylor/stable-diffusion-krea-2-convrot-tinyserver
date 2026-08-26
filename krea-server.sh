#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${KREA_MODEL_ROOT:-/media/aikenyon/WDRed16TB/models/krea}"
SERVER="${SD_SERVER:-${ROOT_DIR}/vendor/stable-diffusion.cpp/build/bin/sd-server}"
DIT="${KREA_DIT:-${MODEL_ROOT}/Krea-2-Raw-INT8-ConvRot/krea-2-raw-int8-convrot.safetensors}"
LLM="${KREA_LLM:-${MODEL_ROOT}/components/text_encoders/Qwen3VL-4B-Instruct-Q4_K_M.gguf}"
VAE="${KREA_VAE:-${MODEL_ROOT}/components/vae/qwen_image_vae.safetensors}"
LORA_DIR="${KREA_LORA_DIR:-${MODEL_ROOT}/krea2_raw_to_turbo_r256_LORA}"
RUNTIME_DIR="${ROOT_DIR}/.runtime"
PID_FILE="${RUNTIME_DIR}/krea-server.pid"
LOG_FILE="${RUNTIME_DIR}/krea-server.log"
HOST="${KREA_SERVER_HOST:-127.0.0.1}"
PORT="${KREA_SERVER_PORT:-1234}"

is_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null
}

case "${1:-status}" in
    start)
        if is_running; then
            echo "Krea server is already running (PID $(<"$PID_FILE"))."
            exit 0
        fi
        mkdir -p "$RUNTIME_DIR"
        for required in "$SERVER" "$DIT" "$LLM" "$VAE"; do
            [[ -f "$required" ]] || { echo "missing required file: $required" >&2; exit 1; }
        done
        nohup "$SERVER" \
            --listen-ip "$HOST" \
            --listen-port "$PORT" \
            --diffusion-model "$DIT" \
            --llm "$LLM" \
            --vae "$VAE" \
            --lora-model-dir "$LORA_DIR" \
            --diffusion-fa \
            --eager-load \
            --steps 52 \
            --cfg-scale 3.5 \
            >"$LOG_FILE" 2>&1 &
        server_pid=$!
        echo "$server_pid" >"$PID_FILE"
        echo "Krea server starting (PID $server_pid)."
        echo "Log: $LOG_FILE"
        echo "Use '$0 logs' to watch loading; status becomes ready after the API responds."
        ;;
    stop)
        if ! is_running; then
            echo "Krea server is not running."
            rm -f "$PID_FILE"
            exit 0
        fi
        server_pid="$(<"$PID_FILE")"
        kill "$server_pid"
        for _ in {1..30}; do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$server_pid" 2>/dev/null; then
            echo "Server did not stop within 30 seconds; PID $server_pid is still running." >&2
            exit 1
        fi
        rm -f "$PID_FILE"
        echo "Krea server stopped; model VRAM has been released."
        ;;
    status)
        if ! is_running; then
            echo "stopped"
            exit 1
        fi
        if curl --silent --fail --max-time 2 "http://${HOST}:${PORT}/sdcpp/v1/capabilities" >/dev/null; then
            echo "ready (PID $(<"$PID_FILE"), http://${HOST}:${PORT})"
        else
            echo "loading (PID $(<"$PID_FILE"))"
        fi
        ;;
    logs)
        touch "$LOG_FILE"
        tail -f "$LOG_FILE"
        ;;
    *)
        echo "usage: $0 {start|stop|status|logs}" >&2
        exit 2
        ;;
esac
