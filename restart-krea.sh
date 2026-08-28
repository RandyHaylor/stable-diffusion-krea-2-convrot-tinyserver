#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "Stopping Krea web UI and tunnel..."
"${ROOT_DIR}/stop-krea-web.sh"

echo "Stopping Krea model server..."
"${ROOT_DIR}/krea-server.sh" stop

echo "Starting Krea model server and web UI..."
"${ROOT_DIR}/start-krea-web.sh" --no-follow

echo "Following Krea server output from ${ROOT_DIR}/.runtime/krea-server.log"
tail -n 80 -F "${ROOT_DIR}/.runtime/krea-server.log"
