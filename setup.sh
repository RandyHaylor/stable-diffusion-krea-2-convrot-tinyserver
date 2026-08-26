#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM_DIR="${ROOT_DIR}/vendor/stable-diffusion.cpp"
UPSTREAM_COMMIT="50d640568388f876b0d63ee6ddb6bc86d997ec64"
CUDA_ARCH="${CUDA_ARCH:-86}"
BUILD_JOBS="${BUILD_JOBS:-2}"

if [[ ! -d "${UPSTREAM_DIR}/.git" ]]; then
    mkdir -p "${ROOT_DIR}/vendor"
    git clone --recurse-submodules https://github.com/leejet/stable-diffusion.cpp.git "$UPSTREAM_DIR"
fi

git -C "$UPSTREAM_DIR" fetch origin "$UPSTREAM_COMMIT"
git -C "$UPSTREAM_DIR" checkout "$UPSTREAM_COMMIT"
git -C "$UPSTREAM_DIR" submodule update --init --recursive
if git -C "$UPSTREAM_DIR" apply --check "${ROOT_DIR}/patches/stable-diffusion-cpp-int8-rowwise.patch" 2>/dev/null; then
    git -C "$UPSTREAM_DIR" apply "${ROOT_DIR}/patches/stable-diffusion-cpp-int8-rowwise.patch"
fi

cmake -S "$UPSTREAM_DIR" -B "${UPSTREAM_DIR}/build" \
    -DSD_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DSD_SERVER_BUILD_FRONTEND=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${UPSTREAM_DIR}/build" --config Release -j "$BUILD_JOBS" --target sd-cli sd-server

echo "Build complete. Download the model files documented in README.md, then run ./krea-server.sh start."
