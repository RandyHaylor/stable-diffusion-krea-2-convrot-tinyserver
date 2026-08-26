# Krea 2 Raw INT8 ConvRot — local generation

This workspace runs the community Krea 2 Raw INT8 ConvRot checkpoint with a
CUDA build of `stable-diffusion.cpp`.

## Generate

```bash
./run-krea.sh "a cinematic photograph of a red fox walking through snow"
```

Both generation scripts use a persistent server. The first request loads the
models; later requests reuse the weights already resident in VRAM:

```bash
./run-krea-turbo.sh "a cinematic photograph of a red fox walking through snow"
```

Each script has `UNLOAD_MODEL_ON_FINISH=false` near the top. Leave it `false`
to keep the model loaded, or change it to `true` to release VRAM automatically
after that script finishes.

Manual server control:

```bash
./krea-server.sh start
./krea-server.sh status
./krea-server.sh logs
./krea-server.sh stop    # unloads the model and releases VRAM
```

Images are written to `outputs/`. Defaults follow the official Raw recipe:
1024×1024, 52 steps, CFG 3.5, seed 0. Override them with environment variables:

```bash
KREA_STEPS=28 KREA_CFG=4.5 KREA_SEED=42 ./run-krea.sh "your prompt"
```

Extra `sd-cli` flags can follow the prompt. For lower VRAM, try:

```bash
./run-krea.sh "your prompt" --offload-to-cpu
```

The initial server start must read roughly 19 GB from the model drive and copy
it to VRAM, so startup takes around two minutes on the current WDRed disk.

## Components

- DiT: `/media/aikenyon/WDRed16TB/models/krea/Krea-2-Raw-INT8-ConvRot/krea-2-raw-int8-convrot.safetensors`
- Text encoder: `/media/aikenyon/WDRed16TB/models/krea/components/text_encoders/Qwen3VL-4B-Instruct-Q4_K_M.gguf`
- VAE: `/media/aikenyon/WDRed16TB/models/krea/components/vae/qwen_image_vae.safetensors`
- Runtime revision: `stable-diffusion.cpp` commit `50d640568388f876b0d63ee6ddb6bc86d997ec64`

The downloaded checkpoint uses the newer ComfyUI format name
`int8_rowwise`. The runtime currently recognizes the same layout only under
its older name, `int8_tensorwise`, so the vendored runtime has a narrow alias
patch in `src/model_io/safetensors_io.cpp`. The underlying I8 weights,
per-output-row scales, ConvRot flag, and H256 group size are unchanged.

## Rebuild

```bash
cmake -S vendor/stable-diffusion.cpp \
  -B vendor/stable-diffusion.cpp/build \
  -DSD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DSD_SERVER_BUILD_FRONTEND=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build vendor/stable-diffusion.cpp/build --config Release -j 2
```

CUDA architecture 86 is pinned for the RTX 3090 Ti. The NVIDIA driver must be
available to generate an image, even though compilation itself only needs the
CUDA toolkit.

## Verified

The complete pipeline was smoke-tested on the RTX 3090 Ti at 256×256 with one
step. It detected Krea2, loaded 192 INT8 ConvRot layers, placed 18.9 GB of
parameters in VRAM, sampled with CUDA, decoded with the Qwen/Wan VAE, and
wrote a valid PNG.
