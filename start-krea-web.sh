#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${ROOT_DIR}/.runtime"
WEB_PID_FILE="${RUNTIME_DIR}/krea-web.pid"
TUNNEL_PID_FILE="${RUNTIME_DIR}/krea-web-tunnel.pid"
WEB_LOG="${RUNTIME_DIR}/krea-web.log"
TUNNEL_LOG="${RUNTIME_DIR}/krea-web-tunnel.log"
CREDS_FILE="${RUNTIME_DIR}/krea-web-credentials"
WEB_PORT="${KREA_WEB_PORT:-7860}"
WEB_USER="${KREA_WEB_USER:-krea}"
FOLLOW_LOG=true
if [[ "${1:-}" == "--no-follow" ]]; then
    FOLLOW_LOG=false
fi

mkdir -p "$RUNTIME_DIR"
if [[ -f "$WEB_PID_FILE" ]] && kill -0 "$(<"$WEB_PID_FILE")" 2>/dev/null; then
    echo "Krea web is already running."
    [[ -f "$CREDS_FILE" ]] && cat "$CREDS_FILE"
else
    WEB_PASSWORD="${KREA_WEB_PASSWORD:-kenyon514krea}"
    printf 'Username: %s\nPassword: %s\nLocal URL: http://127.0.0.1:%s\n' "$WEB_USER" "$WEB_PASSWORD" "$WEB_PORT" >"$CREDS_FILE"
    chmod 600 "$CREDS_FILE"

    if ! "${ROOT_DIR}/krea-server.sh" status >/dev/null 2>&1; then
        "${ROOT_DIR}/krea-server.sh" start
    fi

    nohup python3 "${ROOT_DIR}/scripts/krea_web.py" \
        --host 127.0.0.1 --port "$WEB_PORT" \
        --backend "http://${KREA_SERVER_HOST:-127.0.0.1}:${KREA_SERVER_PORT:-1234}" \
        --user "$WEB_USER" --password "$WEB_PASSWORD" >"$WEB_LOG" 2>&1 &
    echo $! >"$WEB_PID_FILE"

    for _ in {1..60}; do
        curl -sSf "http://127.0.0.1:${WEB_PORT}/health" >/dev/null 2>&1 && break
        sleep 0.25
    done
    if ! curl -sSf "http://127.0.0.1:${WEB_PORT}/health" >/dev/null; then
        kill "$(<"$WEB_PID_FILE")" 2>/dev/null || true
        rm -f "$WEB_PID_FILE"
        echo "Web UI failed to start; see $WEB_LOG" >&2
        tail -n 40 "$WEB_LOG" >&2
        exit 1
    fi

    if [[ "${KREA_WEB_PUBLIC:-true}" == true ]]; then
        nohup "${HOME}/.local/bin/cloudflared" tunnel --url "http://127.0.0.1:${WEB_PORT}" --no-autoupdate >"$TUNNEL_LOG" 2>&1 &
        echo $! >"$TUNNEL_PID_FILE"
        public_url=""
        for _ in {1..80}; do
            public_url="$(sed -n 's#.*\(https://[-a-z0-9]*\.trycloudflare\.com\).*#\1#p' "$TUNNEL_LOG" | tail -1)"
            [[ -n "$public_url" ]] && break
            sleep 0.25
        done
        if [[ -n "$public_url" ]]; then
            printf 'Public URL: %s\n' "$public_url" >>"$CREDS_FILE"
        else
            printf 'Public URL: tunnel starting; watch %s\n' "$TUNNEL_LOG" >>"$CREDS_FILE"
        fi
    fi

    echo "Krea web started."
    cat "$CREDS_FILE"
    echo "Stop the public UI with ./stop-krea-web.sh (the model server stays loaded)."
fi

if [[ "$FOLLOW_LOG" == true ]]; then
    echo "Following web server output from $WEB_LOG"
    tail -n 40 -F "$WEB_LOG"
fi
