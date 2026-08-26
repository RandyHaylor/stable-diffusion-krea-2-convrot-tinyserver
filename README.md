# Krea 2 Raw INT8 ConvRot — local generation

This workspace runs the community Krea 2 Raw INT8 ConvRot checkpoint with a
CUDA build of `stable-diffusion.cpp`.

## Generate

```bash
./run-krea.sh "a cinematic photograph of a red fox walking through snow"
```

The generation scripts use a persistent server. The first request loads the
models; later requests reuse the weights already resident in VRAM:

```bash
./run-krea-turbo.sh "a cinematic photograph of a red fox walking through snow"
```

The Turbo launcher defaults to the tested low-step recipe: LoRA strength 0.6,
6 steps, CFG 1.0, `res_2s`, and the beta57 schedule (`beta` with
`alpha=0.5,beta=0.7`). Every value remains overrideable:

```bash
KREA_LORA_STRENGTH=0.7 KREA_STEPS=8 KREA_CFG=1.0 \
  KREA_SAMPLER=res_2s KREA_SCHEDULER=beta \
  KREA_EXTRA_SAMPLE_ARGS='alpha=0.5,beta=0.7' \
  ./run-krea-turbo.sh "your prompt"
```

For a two-stage Turbo render, generate the composition at 768×768, bilinearly
upscale its latent to 1536×1536, then refine it for three effective steps:

```bash
./run-krea-turbo-highres.sh "your prompt"
```

This is equivalent to:

```bash
KREA_WIDTH=768 KREA_HEIGHT=768 KREA_STEPS=3 \
  KREA_HIRES_WIDTH=1536 KREA_HIRES_HEIGHT=1536 \
  KREA_HIRES_STEPS=3 KREA_HIRES_DENOISE=0.5 \
  ./run-krea-turbo.sh "your prompt"
```

`KREA_HIRES_UPSCALER=Latent` is the runtime's bilinear latent interpolation.
The second pass is an img2img-style refinement: it re-noises the completed
low-resolution latent according to `KREA_HIRES_DENOISE`, rather than literally
continuing one six-step sigma trajectory at its midpoint.

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

For file-manager-friendly control, double-click `start-krea-server.sh` or
`stop-krea-server.sh` and choose **Run** if your file manager asks. The start
launcher waits for Krea to become ready and leaves the model resident after
the window closes.

Images are written to `outputs/`. Defaults follow the official Raw recipe:
1024×1024, 52 steps, CFG 3.5, seed 0. Override them with environment variables:

```bash
KREA_STEPS=28 KREA_CFG=4.5 KREA_SEED=42 ./run-krea.sh "your prompt"
```

Client options can follow the prompt directly. For example:

```bash
./run-krea-turbo.sh "your prompt" --width 768 --height 1024 --seed 42
```

The default model root is the gitignored `models/` directory beside these
scripts. Override it with `KREA_MODEL_ROOT=/some/other/path` if needed. The
first generation reads roughly 19 GB and copies it to VRAM; keeping this folder
on fast local storage materially improves cold-start time.

VAE decoding uses 32×32 **latent-space** tiles by default. Krea's VAE has an 8×
scale factor, so these are 256×256 output-space tiles. This is necessary at
1024×1024 on a 24 GB GPU because the untiled Qwen/Wan VAE graph requests
roughly 7.5 GB of additional VRAM after the model weights are loaded. A value
of 128 or greater spans the complete 1024×1024 latent and effectively disables
tiling. Pass `--no-vae-tiling` only at smaller resolutions or with more VRAM.

## Components

- DiT: `models/Krea-2-Raw-INT8-ConvRot/krea-2-raw-int8-convrot.safetensors`
- Text encoder: `models/components/text_encoders/Qwen3VL-4B-Instruct-Q4_K_M.gguf`
- VAE: `models/components/vae/qwen_image_vae.safetensors`
- Turbo LoRA: `models/krea2_raw_to_turbo_r256_LORA/krea2_raw_to_turbo_r256.safetensors`
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
