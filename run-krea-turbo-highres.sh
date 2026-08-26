#!/usr/bin/env bash
set -euo pipefail

# Two-stage defaults: compose at 768, bilinearly upscale the latent, then
# refine at 1536. Override any value in the environment when desired.
export KREA_WIDTH="${KREA_WIDTH:-768}"
export KREA_HEIGHT="${KREA_HEIGHT:-768}"
export KREA_STEPS="${KREA_STEPS:-3}"
export KREA_HIRES_WIDTH="${KREA_HIRES_WIDTH:-1536}"
export KREA_HIRES_HEIGHT="${KREA_HIRES_HEIGHT:-1536}"
export KREA_HIRES_STEPS="${KREA_HIRES_STEPS:-3}"
export KREA_HIRES_DENOISE="${KREA_HIRES_DENOISE:-0.5}"
export KREA_HIRES_UPSCALER="${KREA_HIRES_UPSCALER:-Latent}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${ROOT_DIR}/run-krea-turbo.sh" "$@"
