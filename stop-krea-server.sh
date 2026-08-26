#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# File-manager launches usually have no terminal. Open one so the result stays
# visible instead of flashing and disappearing.
if [[ "${1:-}" != "--terminal" && ! -t 1 ]]; then
    if command -v x-terminal-emulator >/dev/null 2>&1; then
        exec x-terminal-emulator -e bash -lc "'${ROOT_DIR}/stop-krea-server.sh' --terminal"
    elif command -v gnome-terminal >/dev/null 2>&1; then
        exec gnome-terminal -- bash -lc "'${ROOT_DIR}/stop-krea-server.sh' --terminal"
    elif command -v konsole >/dev/null 2>&1; then
        exec konsole -e bash -lc "'${ROOT_DIR}/stop-krea-server.sh' --terminal"
    fi
fi

echo "Stopping the persistent Krea 2 server..."
"${ROOT_DIR}/krea-server.sh" stop
echo
echo "The server is stopped and its model VRAM has been released."
echo
read -r -p "Press Enter to close this window..." || true
