#!/usr/bin/env bash
set -euo pipefail

# The native hires API only returns the final image. When true, first perform a
# matching low-resolution render and save it with a distinct filename.
SAVE_LOWRES_IMAGE=true

# Two-stage defaults: compose at 768, bilinearly upscale the latent, then
# refine at 1536. Override any value in the environment when desired.
export KREA_WIDTH="${KREA_WIDTH:-1024}"
export KREA_HEIGHT="${KREA_HEIGHT:-1024}"
export KREA_STEPS="${KREA_STEPS:-6}"
export KREA_HIRES_WIDTH="${KREA_HIRES_WIDTH:-1536}"
export KREA_HIRES_HEIGHT="${KREA_HIRES_HEIGHT:-1536}"
export KREA_HIRES_STEPS="${KREA_HIRES_STEPS:-6}"
export KREA_HIRES_DENOISE="${KREA_HIRES_DENOISE:-0.5}"
export KREA_HIRES_UPSCALER="${KREA_HIRES_UPSCALER:-Latent}"

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$SAVE_LOWRES_IMAGE" == true ]]; then
    echo "Saving the matching ${KREA_WIDTH}x${KREA_HEIGHT} low-resolution render first..."
    KREA_HIRES_WIDTH=0 \
    KREA_HIRES_HEIGHT=0 \
    KREA_OUTPUT_PREFIX=krea-turbo-lowres \
        "${ROOT_DIR}/run-krea-turbo.sh" "$@"
fi

echo "Rendering the ${KREA_HIRES_WIDTH}x${KREA_HIRES_HEIGHT} hires result..."
KREA_OUTPUT_PREFIX="${KREA_OUTPUT_PREFIX:-krea-turbo-highres}" \
    exec "${ROOT_DIR}/run-krea-turbo.sh" "$@"
