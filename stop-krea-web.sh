#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for name in krea-web-tunnel krea-web; do
    pid_file="${ROOT_DIR}/.runtime/${name}.pid"
    if [[ -f "$pid_file" ]]; then
        pid="$(<"$pid_file")"
        kill "$pid" 2>/dev/null || true
        rm -f "$pid_file"
    fi
done
echo "Krea web UI and public tunnel stopped. The Krea model server was left running."
