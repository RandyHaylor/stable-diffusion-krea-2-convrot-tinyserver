#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# File-manager launches usually have no terminal. Open one so startup progress
# and any errors remain visible.
if [[ "${1:-}" != "--terminal" && ! -t 1 ]]; then
    if command -v x-terminal-emulator >/dev/null 2>&1; then
        exec x-terminal-emulator -e bash -lc "'${ROOT_DIR}/start-krea-server.sh' --terminal"
    elif command -v gnome-terminal >/dev/null 2>&1; then
        exec gnome-terminal -- bash -lc "'${ROOT_DIR}/start-krea-server.sh' --terminal"
    elif command -v konsole >/dev/null 2>&1; then
        exec konsole -e bash -lc "'${ROOT_DIR}/start-krea-server.sh' --terminal"
    fi
fi

echo "Starting the persistent Krea 2 server..."
"${ROOT_DIR}/krea-server.sh" start
echo

for _ in {1..240}; do
    status="$(${ROOT_DIR}/krea-server.sh status 2>/dev/null || true)"
    printf '\r%-78s' "Status: ${status}"
    if [[ "$status" == ready* ]]; then
        echo
        echo
        echo "Krea HTTP server is ready. Model loading is lazy: the first"
        echo "generation loads the weights; its terminal will show progress."
        echo "After that, the model remains loaded in VRAM."
        echo "Run run-krea.sh or run-krea-turbo.sh to generate images."
        echo "Run krea-server.sh stop when you want to release VRAM."
        echo
        echo "Following server output from ${ROOT_DIR}/.runtime/krea-server.log"
        tail -n 80 -F "${ROOT_DIR}/.runtime/krea-server.log"
        exit 0
    fi
    if [[ "$status" == stopped* ]]; then
        echo
        echo
        echo "The server exited before becoming ready. Recent log output:"
        tail -80 "${ROOT_DIR}/.runtime/krea-server.log" 2>/dev/null || true
        echo
        exit 1
    fi
    sleep 1
done

echo
echo
echo "Timed out waiting for the server. Inspect .runtime/krea-server.log."
tail -n 80 "${ROOT_DIR}/.runtime/krea-server.log" 2>/dev/null || true
exit 1
