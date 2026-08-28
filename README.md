# Krea 2 Raw INT8 ConvRot — local generation

This workspace runs the community Krea 2 Raw INT8 ConvRot checkpoint with a
CUDA build of `stable-diffusion.cpp`.

## Generate

```bash
./run-krea.sh "a cinematic photograph of a red fox walking through snow"
```

The generation scripts use a persistent server. The first request loads the
models; later requests reuse the weights already resident in VRAM.

The web UI is the primary way to generate. Any LoRA placed in `models/loras`
is listed there and applied only when you select it, at the strength you set —
including the raw-to-turbo LoRA, which the app treats no differently from any
other. Low-step Turbo output is a matter of selecting that LoRA and setting the
sampling fields accordingly; a tested recipe is LoRA strength 0.6, 6 steps and
CFG 1.0. The default sampler and scheduler are `er_sde` and `discrete`.

For a two-stage render, generate the composition at 768×768, bilinearly upscale
its latent to 1536×1536, then refine it for three effective steps. Enable
`Hires pass` in the UI and set the hires resolution, steps, and denoise.
`Latent` is the runtime's bilinear latent interpolation.
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
./run-krea.sh "your prompt" --width 768 --height 1024 --seed 42
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

## Web UI

```bash
./start-krea-web.sh    # starts the Krea server if needed, then the authenticated UI
./stop-krea-web.sh     # closes the UI and public link; the model stays loaded
```

The launcher prints a generated username/password and the local URL, and also
writes them to `.runtime/krea-web-credentials` with mode 600. Set
`KREA_WEB_USER` and `KREA_WEB_PASSWORD` to pin your own credentials, or
`KREA_WEB_PORT` to move off 7860.

By default the launcher also opens a Cloudflare Quick Tunnel and appends the
temporary `https://<random>.trycloudflare.com` URL to the credentials file. That
URL is publicly reachable, so HTTP Basic authentication in the app is the only
access control — Quick Tunnels are intended for temporary development use. Set
`KREA_WEB_PUBLIC=false` to stay local-only.

The inference server itself stays bound to `127.0.0.1`; only the UI is exposed.

Every field in the UI is a free-text input with a datalist of typical values,
including common SDXL resolutions and the 1248×1824 portrait family, so the
suggestions are a starting point rather than a constraint. VAE tile
size is exposed with 32/64/68/128/256/512; those are **latent** units, so they
correspond to 256/512/544/1024/2048/4096 output pixels. Larger values are the
usual cause of an out-of-memory decode at 2048.

PAG is available for Krea 2 through **Enable PAG**, with scale, comma-separated
zero-based transformer layers, and an inclusive 0–1 sampling window. A useful
starting point is scale 0.5–1.5 on one layer (the UI starts at layer 7). PAG adds
one conditional model evaluation at each active sampling step. It combines with
CFG as `cfg_prediction + pag_scale * (conditional - perturbed)` and is
deliberately disabled for the native hires refinement stage.

Generation is serialized through a single worker: one request runs at a time
and the rest wait. The controls are **Queue generation**, **Queue in front**,
per-job **Remove**, **Clear waiting**, **Kill current**, and **Kill all**
(cancel the running job and clear the queue). The header shows the queue count.

"Kill current" needs the vendored `/sdcpp/v1/cancel` endpoint, which invokes the
runtime's existing atomic cancellation flag. It interrupts sampling without
killing the process or unloading the weights. A server started from a binary
built before that patch answers 404 and only the queue controls will work —
restart the server after rebuilding.

## Components

- DiT: `models/chkpt/krea2RawBaseInt8Row_v10.safetensors`
  (override with `KREA_DIT=`; `models/chkpt/krea-2-raw-int8-convrot.safetensors`
  is the previous checkpoint and still loads)
- Text encoder: `models/text/Qwen3VL-4B-Instruct-Q4_K_M.gguf`
- VAE: `models/vae/qwen_image_vae.safetensors`
- LoRAs: any `*.safetensors` in `models/loras`, selectable in the web UI
- Runtime revision: `stable-diffusion.cpp` commit `50d640568388f876b0d63ee6ddb6bc86d997ec64`

The downloaded checkpoint uses the newer ComfyUI format name
`int8_rowwise`. The runtime currently recognizes the same layout only under
its older name, `int8_tensorwise`, so the vendored runtime has a narrow alias
patch in `src/model_io/safetensors_io.cpp`. The underlying I8 weights,
per-output-row scales, ConvRot flag, and H256 group size are unchanged.

`patches/stable-diffusion-cpp-int8-rowwise.patch` carries that alias, the
`/sdcpp/v1/cancel` endpoint, and the Krea 2 PAG runtime/API extension. Regenerate
it from the vendor tree rather than editing it by hand:

```bash
git -C vendor/stable-diffusion.cpp diff > patches/stable-diffusion-cpp-int8-rowwise.patch
```

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

The web queue and its controls are covered by a GPU-free test that runs the
real app against a stub backend:

```bash
python3 scripts/test_krea_web_queue_controls.py
```

It asserts authentication, one-at-a-time execution, front-insertion ordering,
removing a waiting job, refusing to "remove" a running one, cancel-current
reaching the backend, kill-all draining the queue, job-id-namespaced output
filenames, authenticated downloads, PAG request serialization, and
path-traversal rejection.

Not yet verified end-to-end against the loaded model: PAG image output, "Kill
current" interrupting a real sampling run, and whether 1024 → 2048 hires fits
in 24 GB.

A 1024 → 2048 hires attempt failed with the DiT compute buffer, not the VAE:
after `hires Latent upscale 128x128 -> 256x256` the runtime asked for 4635 MiB
and `krea2 alloc compute buffer failed`. Decoding never ran, so VAE tile size
does not govern that failure. The comparable 1536 pass asks for 2689 MiB and
has both succeeded and failed in the same session depending on free VRAM.
